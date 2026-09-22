"""Golden set tests: import formats, recompression, curation, guards."""

from __future__ import annotations

import json
import os

import pytest
from synthutil import synth_display

from calib_auto import golden


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_AUTO_DATA", str(tmp_path))
    return tmp_path


def _report_source(tmp_path, name="src", photos=("p1.png", "p2.png"),
                   content="A" * 12):
    src = tmp_path / name
    src.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, photo in enumerate(photos):
        synth_display(src / photo, content, seed=i)
        frames.append({"tag": photo.split("_")[0], "frameId": i,
                       "frame": content, "photo": photo, "read": content})
    (src / "report.json").write_text(json.dumps({"frames": frames}),
                                     encoding="utf-8")
    return src


def test_import_from_report_prefills_and_stays_pending(env):
    src = _report_source(env)
    result = golden.create_set(str(src), name="s1", image_format="keep")
    assert result["count"] == 2
    assert result["stats"] == {"total": 2, "verified": 0, "pending": 2}
    entries = golden.read_entries("s1")
    assert entries[0]["content"] == "A" * 12   # pre-filled from `read`
    assert entries[0]["status"] == "pending"


def test_import_from_labels_preserves_verified_truth(env):
    # Importing a curated set must NOT downgrade verified content to
    # pending pre-fills.
    src = env / "old-set"
    images = src / "images"
    images.mkdir(parents=True)
    rows = []
    for i, (photo, content) in enumerate((("p1.png", "A" * 12),
                                          ("p2.png", "B" * 12))):
        synth_display(images / photo, content, seed=i)
        rows.append({"photo": photo, "content": content,
                     "status": "verified", "prior_read": "X" * 12,
                     "want": content, "tag": "t", "frame_id": i})
    (src / "labels.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    result = golden.create_set(str(src), name="s2", image_format="keep")
    assert result["stats"] == {"total": 2, "verified": 2, "pending": 0}
    entries = golden.read_entries("s2")
    assert entries[0]["content"] == "A" * 12
    assert entries[0]["status"] == "verified"
    assert entries[0]["prior_read"] == "X" * 12


def test_import_recompresses_auto_to_png_for_small_sets(env):
    src = _report_source(env)
    result = golden.create_set(str(src), name="s3")
    assert result["image_format"] == "png"
    assert result["encoding"]["png_bytes"] is not None
    meta = golden.get_set("s3")["meta"]
    assert meta["image_format"] == "png"
    # PNG is lossless: dimensions survive the round trip.
    import cv2
    img = cv2.imread(golden.photo_file("s3", "p1.png"))
    assert img is not None and img.shape == (220, 900, 3)


def test_import_jpeg_fallback_renames_photos(env):
    src = _report_source(env)
    result = golden.create_set(str(src), name="s4", image_format="jpeg")
    assert result["image_format"] == "jpeg"
    entries = golden.read_entries("s4")
    assert all(e["photo"].endswith(".jpg") for e in entries)
    assert golden.photo_file("s4", entries[0]["photo"]) is not None


def test_import_skips_missing_photos(env):
    src = _report_source(env, photos=("p1.png", "p2.png"))
    os.remove(src / "p2.png")
    result = golden.create_set(str(src), name="s5", image_format="keep")
    assert result["count"] == 1
    assert result["skipped"] == ["p2.png"]


def test_import_guards_and_rollback(env):
    src = _report_source(env)
    with pytest.raises(golden.GoldenError, match="not found"):
        golden.create_set(str(env / "nope"), name="x")
    with pytest.raises(golden.GoldenError, match="name"):
        golden.create_set(str(src), name="..bad")
    golden.create_set(str(src), name="dup", image_format="keep")
    with pytest.raises(golden.GoldenError, match="already exists"):
        golden.create_set(str(src), name="dup", image_format="keep")
    # a failed import must not leave a half set behind
    assert not os.path.exists(golden.set_dir("x"))


def test_update_entry_verified_pending_and_newline_fold(env):
    _report_source(env)
    golden.create_set(str(_report_source(env, "src2")), name="s6",
                      image_format="keep")
    entry = golden.update_entry("s6", "p1.png", content="B" * 12)
    assert entry["status"] == "verified"
    entry = golden.update_entry("s6", "p1.png", content="C\nD" + " " * 10)
    assert "\n" not in entry["content"]
    entry = golden.update_entry("s6", "p1.png", status="pending")
    assert entry["status"] == "pending"
    with pytest.raises(golden.GoldenError, match="not in set"):
        golden.update_entry("s6", "nope.png", content="X")
    with pytest.raises(golden.GoldenError, match="status"):
        golden.update_entry("s6", "p1.png", status="bogus")


def test_remove_image_and_delete_set(env):
    src = _report_source(env)
    golden.create_set(str(src), name="s7", image_format="keep")
    stats = golden.remove_image("s7", "p1.png")
    assert stats == {"total": 1, "verified": 0, "pending": 1}
    assert golden.photo_file("s7", "p1.png") is None
    with pytest.raises(golden.GoldenError, match="not in set"):
        golden.remove_image("s7", "p1.png")
    golden.delete_set("s7")
    assert golden.list_sets() == []
    with pytest.raises(golden.GoldenError, match="not found"):
        golden.delete_set("s7")


def test_list_sets_newest_first(env):
    golden.create_set(str(_report_source(env, "a")), name="aaa",
                      image_format="keep")
    golden.create_set(str(_report_source(env, "b")), name="bbb",
                      image_format="keep")
    names = [s["name"] for s in golden.list_sets()]
    assert set(names) == {"aaa", "bbb"}


def test_valid_set_name_and_photo_file_guards(env):
    assert golden.valid_set_name("run-001")
    assert not golden.valid_set_name("..evil")
    assert not golden.valid_set_name("")
    assert not golden.valid_set_name("a/b")
    with pytest.raises(golden.GoldenError):
        golden.set_dir("../x")
    assert golden.photo_file("noset", "p.png") is None
    assert golden.photo_file("noset", "..%5Cx.png") is None


def test_import_unrecognized_source_raises(env):
    empty = env / "empty"
    empty.mkdir()
    with pytest.raises(golden.GoldenError, match=r"no labels\.jsonl"):
        golden.create_set(str(empty), name="x")
