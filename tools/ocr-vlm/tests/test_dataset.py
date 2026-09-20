"""Dataset tests: parsing, validation markers, photo guard, discovery."""

from __future__ import annotations

import json
import os

from ocr_vlm.dataset import (ISSUE_DUPLICATE, ISSUE_PHOTO, describe,
                             discover, load_dataset, photo_path, prefix_of,
                             valid_photo_name)

W12 = "A" * 12


def _write_jsonl(directory, lines, name="reads.jsonl"):
    path = os.path.join(str(directory), name)
    with open(path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")
    return path


def _touch(directory, name):
    with open(os.path.join(str(directory), name), "w", encoding="utf-8"):
        pass


def test_load_valid(tmp_path):
    _touch(tmp_path, "sw_1_f1.png")
    _touch(tmp_path, "sw_2_f2.png")
    _write_jsonl(tmp_path, [
        json.dumps({"photo": "sw_1_f1.png", "want": W12, "saw": W12}),
        json.dumps({"photo": "sw_2_f2.png", "want": W12, "saw": "A" * 11 + "X"}),
    ])
    ds = load_dataset(str(tmp_path))
    assert len(ds.records) == 2
    assert [r.index for r in ds.records] == [0, 1]
    assert [r.photo for r in ds.records] == ["sw_1_f1.png", "sw_2_f2.png"]
    assert all(not r.fatal and not r.issues for r in ds.records)
    assert ds.problems == [] and ds.missing_photos == []
    assert ds.records[0].width == 12

    d = describe(ds)
    assert d["rows"] == 2 and d["runnable"] == 2
    assert d["problems_count"] == 0 and d["missing_count"] == 0
    assert d["flagged_count"] == 0
    assert d["widths"] == {"min": 12, "max": 12}
    assert d["prefixes"] == [{"prefix": "sw", "rows": 2}]


def test_load_marks_bad_rows_instead_of_dropping(tmp_path):
    _touch(tmp_path, "ok_1.png")
    _touch(tmp_path, "nosaw_3.png")
    _write_jsonl(tmp_path, [
        "{not json",                                                        # 1
        json.dumps({"photo": "ok_1.png", "want": W12, "saw": W12}),         # 2 ok
        json.dumps({"photo": "missing_2.png", "want": W12, "saw": W12}),    # 3 missing photo
        json.dumps({"want": W12}),                                          # 4 no photo
        json.dumps({"photo": "ok_1.png", "want": "AAAA",
                    "saw": "AAAA"}),                                        # 5 dupe + width
        json.dumps({"photo": "nosaw_3.png", "want": W12}),                  # 6 no saw
        json.dumps([1, 2, 3]),                                              # 7 not an object
        json.dumps({"photo": "bad/../x.png", "want": "A",
                    "saw": "A"}),                                           # 8 bad name
    ])
    ds = load_dataset(str(tmp_path))
    assert len(ds.records) == 6                     # one per JSON object line
    assert len(ds.problems) == 2                     # lines 1 and 7
    assert any("line 1" in p for p in ds.problems)

    by_index = {r.index: r for r in ds.records}
    assert by_index[0].runnable                        # line 2
    assert ISSUE_PHOTO in by_index[1].issues           # line 3
    assert ds.missing_photos == ["missing_2.png"]
    assert "missing photo name" in by_index[2].fatal[0]  # line 4
    assert any(i.startswith(ISSUE_DUPLICATE)             # line 5
               for i in by_index[3].issues)
    assert any("width 4" in i for i in by_index[3].issues)
    assert "no saw baseline" in by_index[4].issues       # line 6
    assert "bad photo name" in by_index[5].fatal[0]      # line 8

    d = describe(ds)
    assert d["rows"] == 6 and d["runnable"] == 4
    assert d["problems_count"] == 2
    assert d["missing_count"] == 1
    assert d["flagged_count"] == 5
    assert d["widths"] == {"min": 1, "max": 12}


def test_load_unreadable_file(tmp_path):
    ds = load_dataset(str(tmp_path), "nope.jsonl")
    assert ds.records == []
    assert ds.problems and "cannot read" in ds.problems[0]


def test_photo_name_guard():
    assert valid_photo_name("sw_1_f1.png")
    assert valid_photo_name("a.JPG")
    assert not valid_photo_name("../x.png")
    assert not valid_photo_name("a\\b.png")
    assert not valid_photo_name("a/b.png")
    assert not valid_photo_name(".hidden.png")
    assert not valid_photo_name("a.txt")
    assert not valid_photo_name("")


def test_photo_path_resolution(tmp_path):
    _touch(tmp_path, "ok.png")
    assert photo_path(str(tmp_path), "ok.png")
    assert photo_path(str(tmp_path), "nope.png") is None
    assert photo_path(str(tmp_path), "..\\ok.png") is None
    assert photo_path(str(tmp_path), "..") is None


def test_prefix_of():
    assert prefix_of("sw_37_f255.png") == "sw"
    assert prefix_of("ladder_t0_f345.png") == "ladder"
    assert prefix_of("p1r_32_f300.png") == "p1r"
    assert prefix_of("plain.png") == "plain"
    assert prefix_of("_weird.png") == "other"


def test_discover(tmp_path):
    tools = tmp_path / "tools"
    for parts in (("calib-vlm", "data", "runs", "run-001"),
                  ("calib-vlm", "data", "runs", "run-002"),
                  ("calib-agent", "app", "data", "runs", "run-777")):
        d = tools.joinpath(*parts)
        d.mkdir(parents=True, exist_ok=True)
        (d / "reads.jsonl").write_text("", encoding="utf-8")
    (tools / "calib-vlm" / "data" / "runs" / "run-003").mkdir(
        parents=True)  # run dir without a dataset
    found = discover(tools_dir=str(tools))
    assert found == [
        str(tools / "calib-vlm" / "data" / "runs" / "run-002"),
        str(tools / "calib-vlm" / "data" / "runs" / "run-001"),
        str(tools / "calib-agent" / "app" / "data" / "runs" / "run-777"),
    ]
    assert discover(filename="other.jsonl", tools_dir=str(tools)) == []
    assert discover(tools_dir=str(tmp_path / "missing")) == []


# -- baseline sets (curated ground truth) -------------------------------------

def _baseline_set(directory, rows, photos=True):
    """A set folder: baseline.jsonl + images/<photo>."""
    images = directory / "images"
    images.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row) for row in rows]
    (directory / "baseline.jsonl").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")
    if photos:
        for row in rows:
            _touch(images, row["photo"])
    return directory


def _baseline_rows():
    return [
        {"photo": "a.png", "content": "A" * 12, "status": "verified",
         "prior_read": "A" * 11 + "B", "want": "A" * 12},
        {"photo": "b.png", "content": "B" * 12, "status": "verified",
         "prior_read": "", "want": "B" * 12},
        {"photo": "c.png", "content": "C" * 12, "status": "pending",
         "prior_read": "C" * 12, "want": "C" * 12},
    ]


def test_baseline_verified_only_default(tmp_path):
    _baseline_set(tmp_path, _baseline_rows())
    ds = load_dataset(str(tmp_path))          # default name auto-resolves

    assert ds.kind == "baseline"
    assert ds.filename == "baseline.jsonl"
    assert len(ds.records) == 2               # pending excluded
    assert (ds.verified, ds.pending, ds.skipped_pending) == (2, 1, 1)
    assert ds.records[0].want == "A" * 12     # content is the truth
    assert ds.records[0].saw == "A" * 11 + "B"   # prior_read is the baseline
    assert ds.records[1].issues == []         # empty prior_read is fine here

    d = describe(ds)
    assert d["kind"] == "baseline"
    assert (d["verified"], d["pending"], d["skipped_pending"]) == (2, 1, 1)
    assert d["rows"] == 2 and d["runnable"] == 2


def test_baseline_include_pending_flags_unverified(tmp_path):
    _baseline_set(tmp_path, _baseline_rows())
    ds = load_dataset(str(tmp_path), verified_only=False)

    assert len(ds.records) == 3
    assert ds.skipped_pending == 0
    pending_row = [r for r in ds.records if r.photo == "c.png"][0]
    assert any("unverified" in issue for issue in pending_row.issues)


def test_baseline_resolve_prefers_existing_name(tmp_path):
    from ocr_vlm.dataset import resolve_dataset_file

    (tmp_path / "baseline.jsonl").write_text("", encoding="utf-8")
    assert resolve_dataset_file(str(tmp_path),
                                "reads.jsonl") == "baseline.jsonl"
    (tmp_path / "reads.jsonl").write_text("", encoding="utf-8")
    assert resolve_dataset_file(str(tmp_path), "reads.jsonl") == "reads.jsonl"
    assert resolve_dataset_file(str(tmp_path), "other.jsonl") == "other.jsonl"


def test_baseline_missing_photo_flagged(tmp_path):
    rows = [{"photo": "ghost.png", "content": "A" * 12,
             "status": "verified", "prior_read": "", "want": ""}]
    _baseline_set(tmp_path, rows, photos=False)
    ds = load_dataset(str(tmp_path))

    assert ISSUE_PHOTO in ds.records[0].issues
    assert ds.missing_photos == ["ghost.png"]


def test_photo_path_finds_images_subdir(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    _touch(images, "x.png")
    found = photo_path(str(tmp_path), "x.png")
    assert found is not None
    assert os.path.dirname(found) == str(images)
    assert photo_path(str(tmp_path), "nope.png") is None


def test_discover_finds_baseline_sets(tmp_path):
    tools = tmp_path / "tools"
    setdir = tools / "ocr-vlm" / "baselines" / "run-001"
    setdir.mkdir(parents=True)
    (setdir / "baseline.jsonl").write_text("", encoding="utf-8")
    found = discover(filename="baseline.jsonl", tools_dir=str(tools))
    assert found == [str(setdir)]
