"""Baseline-set curation tests (create/read/update/remove/delete)."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

from ocr_vlm import curate


def _png(path, w: int = 32, h: int = 12):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


@pytest.fixture
def baselines(tmp_path, monkeypatch):
    """Point the curation root at a temp folder."""
    root = tmp_path / "baselines"
    monkeypatch.setenv("OCR_VLM_BASELINES", str(root))
    return root


def _source_run(tmp_path, frames, missing=()):
    """A minimal calib-vlm style run dir: report.json frames + photos."""
    src = tmp_path / "run-009"
    src.mkdir(parents=True, exist_ok=True)
    for photo in frames:
        if photo not in missing:
            _png(src / photo)
    payload = {"frames": [
        {"tag": photo.split("_")[0], "frameId": 100 + i,
         "frame": "A" * 12, "photo": photo,
         "read": "A" * 12 if i == 1 else "  B         "}
        for i, photo in enumerate(frames, 1)]}
    (src / "report.json").write_text(json.dumps(payload), encoding="utf-8")
    return src


def test_create_set_from_report_prefills_and_copies(baselines, tmp_path):
    src = _source_run(tmp_path, ["a_f1.png", "b_f2.png"])
    res = curate.create_set(str(src))            # name defaults to dir name

    assert res["name"] == "run-009"
    assert res["count"] == 2
    assert res["skipped"] == []
    assert res["stats"] == {"total": 2, "verified": 0, "pending": 2}

    info = curate.get_set("run-009")
    assert info["meta"]["source"] == str(src)
    assert info["meta"]["module_count"] == 12
    first = info["entries"][0]
    assert first["photo"] == "a_f1.png"
    assert first["content"] == "A" * 12          # prefilled from prior read
    assert first["prior_read"] == "A" * 12
    assert first["want"] == "A" * 12
    assert first["tag"] == "a"
    assert first["frame_id"] == 101
    assert first["status"] == "pending"
    # copied byte-identical, originals untouched
    copied = baselines / "run-009" / "images" / "a_f1.png"
    assert copied.read_bytes() == (src / "a_f1.png").read_bytes()


def test_create_set_from_reads_jsonl(baselines, tmp_path):
    src = tmp_path / "ds"
    src.mkdir()
    _png(src / "x.png")
    (src / "reads.jsonl").write_text(json.dumps(
        {"photo": "x.png", "want": "C" * 12, "saw": "C" * 11 + "D"}) + "\n",
        encoding="utf-8")
    res = curate.create_set(str(src), name="imported")
    entry = curate.get_set("imported")["entries"][0]
    assert res["count"] == 1
    assert entry["content"] == "C" * 11 + "D"   # prefilled from saw
    assert entry["prior_read"] == "C" * 11 + "D"
    assert entry["want"] == "C" * 12
    assert entry["tag"] == ""


def test_create_skips_missing_photos(baselines, tmp_path):
    src = _source_run(tmp_path, ["a_f1.png", "ghost_f9.png"],
                      missing=("ghost_f9.png",))
    res = curate.create_set(str(src))
    assert res["count"] == 1
    assert res["skipped"] == ["ghost_f9.png"]
    names = [e["photo"] for e in curate.get_set("run-009")["entries"]]
    assert names == ["a_f1.png"]


def test_create_guards_and_rollback(baselines, tmp_path):
    src = _source_run(tmp_path, ["a_f1.png"])
    curate.create_set(str(src))
    with pytest.raises(curate.CurateError, match="already exists"):
        curate.create_set(str(src))
    with pytest.raises(curate.CurateError, match="set name must be"):
        curate.create_set(str(src), name="../evil")
    with pytest.raises(curate.CurateError, match="source directory not found"):
        curate.create_set(str(tmp_path / "nope"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(curate.CurateError, match="no reads.jsonl or report"):
        curate.create_set(str(empty), name="empty-set")
    assert not (baselines / "empty-set").exists()

    # all photos missing -> error and no half-created folder left behind
    allgone = _source_run(tmp_path, ["z_f3.png"], missing=("z_f3.png",))
    allgone = allgone.rename(tmp_path / "run-010")
    with pytest.raises(curate.CurateError, match="every source photo"):
        curate.create_set(str(allgone))
    assert not (baselines / "run-010").exists()


def test_update_entry_marks_verified_and_back(baselines, tmp_path):
    src = _source_run(tmp_path, ["a_f1.png"])
    curate.create_set(str(src))

    entry = curate.update_entry("run-009", "a_f1.png", content="Z" * 12)
    assert entry["status"] == "verified"
    assert curate.get_set("run-009")["stats"] == {
        "total": 1, "verified": 1, "pending": 0}

    entry = curate.update_entry("run-009", "a_f1.png", status="pending")
    assert entry["status"] == "pending"
    assert entry["content"] == "Z" * 12          # untouched by a status set

    # newlines are folded: the display content is a single row
    entry = curate.update_entry("run-009", "a_f1.png", content="AB\nCD")
    assert entry["content"] == "AB CD"

    with pytest.raises(curate.CurateError, match="not in set"):
        curate.update_entry("run-009", "nope.png", content="X")
    with pytest.raises(curate.CurateError, match="status must be"):
        curate.update_entry("run-009", "a_f1.png", status="bogus")


def test_remove_image(baselines, tmp_path):
    src = _source_run(tmp_path, ["a_f1.png", "b_f2.png"])
    curate.create_set(str(src))
    stats = curate.remove_image("run-009", "a_f1.png")
    assert stats == {"total": 1, "verified": 0, "pending": 1}
    assert not (baselines / "run-009" / "images" / "a_f1.png").exists()
    assert [e["photo"] for e in curate.get_set("run-009")["entries"]] == \
        ["b_f2.png"]
    with pytest.raises(curate.CurateError, match="not in set"):
        curate.remove_image("run-009", "a_f1.png")


def test_delete_set_and_list_sets(baselines, tmp_path):
    src = _source_run(tmp_path, ["a_f1.png"])
    curate.create_set(str(src), name="one")
    curate.create_set(str(src), name="two")
    assert {s["name"] for s in curate.list_sets()} == {"one", "two"}

    curate.delete_set("one")
    assert {s["name"] for s in curate.list_sets()} == {"two"}
    with pytest.raises(curate.CurateError, match="not found"):
        curate.delete_set("one")


def test_list_sets_ignores_unrelated_dirs(baselines):
    (baselines / "notaset").mkdir(parents=True)
    (baselines / "notaset" / "random.txt").write_text("x", encoding="utf-8")
    (baselines / "bad name").mkdir(parents=True)   # invalid: space
    (baselines / "bad name" / curate.ENTRIES_FILE).write_text(
        "", encoding="utf-8")
    assert curate.list_sets() == []


def test_photo_file_guards(baselines, tmp_path):
    src = _source_run(tmp_path, ["a_f1.png"])
    curate.create_set(str(src))
    assert curate.photo_file("run-009", "a_f1.png") is not None
    assert curate.photo_file("run-009", "nope.png") is None
    assert curate.photo_file("run-009", "..\\a_f1.png") is None
    assert curate.photo_file("run-009", "a/b.png") is None


@pytest.mark.parametrize("name,ok", [
    ("run-001", True), ("a.b_c-d", True), ("A1", True),
    ("", False), (".", False), (".hidden", False),
    ("a/b", False), ("a\\b", False), ("x" * 65, False), ("-lead", False),
])
def test_valid_set_name(name, ok):
    assert curate.valid_set_name(name) is ok


def test_baselines_root_env_override(baselines):
    assert curate.baselines_root() == str(baselines)
