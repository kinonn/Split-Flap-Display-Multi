"""Glyph cache tests: extraction from curated sets, skips, roundtrip."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from ocr_vlm import curate, glyphs

from synthutil import CHUNKY, make_baseline_set, synth_display


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_VLM_BASELINES", str(tmp_path / "baselines"))
    monkeypatch.setenv("OCR_VLM_DATA", str(tmp_path / "data"))
    return tmp_path


def test_extract_set_labels_positions_and_crops(env):
    make_baseline_set(env, "s1", {"p1.png": CHUNKY, "p2.png": CHUNKY})
    payload = glyphs.extract_set("s1")
    summary = payload["summary"]
    assert summary["cells"] == 24
    assert summary["photos"] == 2
    assert summary["skipped"] == 0
    labels = [str(x) for x in payload["labels"]]
    assert labels[:12] == list(CHUNKY)
    assert payload["positions"].tolist()[:12] == list(range(12))
    assert payload["crops"].shape == (24, glyphs.GLYPH_SIZE, glyphs.GLYPH_SIZE)
    assert payload["crops"].dtype == np.uint8
    assert not payload["blanks"].any()          # every cell has a glyph
    assert set(summary["counts"]) == set(CHUNKY)


def test_extract_blank_cells_become_zero_rasters(env):
    make_baseline_set(env, "s2", {"p1.png": "AB          "})
    payload = glyphs.extract_set("s2")
    labels = [str(x) for x in payload["labels"]]
    assert labels.count(" ") == 10
    for i, label in enumerate(labels):
        if label == " ":
            assert not payload["crops"][i].any()   # no ink -> zero raster
            assert bool(payload["blanks"][i])


def test_extract_set_skips_with_reasons(env):
    make_baseline_set(env, "s3", {"ok.png": CHUNKY, "gone.png": CHUNKY,
                                  "bad.png": CHUNKY})
    images = env / "baselines" / "s3" / "images"
    os.remove(images / "gone.png")
    (images / "bad.png").write_bytes(b"definitely not a png")
    curate.update_entry("s3", "ok.png", content=CHUNKY[:5])  # width mismatch
    payload = glyphs.extract_set("s3")
    reasons = payload["summary"]["skipped_reasons"]
    assert payload["summary"]["cells"] == 0
    assert reasons["ok.png"].startswith("content width")
    assert reasons["gone.png"] == "photo file missing"
    assert reasons["bad.png"] == "photo unreadable"


def test_extract_set_rejects_unknown_style(env):
    make_baseline_set(env, "s1", {"p1.png": CHUNKY})
    with pytest.raises(ValueError, match="style"):
        glyphs.extract_set("s1", style="sparkle")


def test_cache_save_load_roundtrip(env):
    make_baseline_set(env, "s4", {"p1.png": CHUNKY})
    payload = glyphs.extract_set("s4")
    npz_path, json_path = glyphs.save_cache(None, payload)
    assert os.path.isfile(npz_path) and os.path.isfile(json_path)
    loaded = glyphs.load_cache(None, "s4")
    assert [str(x) for x in loaded["labels"]] == \
        [str(x) for x in payload["labels"]]
    assert np.array_equal(loaded["crops"], payload["crops"])
    assert loaded["positions"].tolist() == payload["positions"].tolist()
    assert loaded["style"] == "none"
    summary = json.loads(open(json_path, encoding="utf-8").read())
    assert summary["cells"] == 12
    assert summary["set"] == "s4"


def test_available_sets_excludes_bank_artifact(env):
    make_baseline_set(env, "s5", {"p1.png": CHUNKY})
    glyphs.save_cache(None, glyphs.extract_set("s5"))
    assert glyphs.available_sets(None) == ["s5"]
    # the trained bank lives in the same folder and must not be listed
    # as a crop-cache set (its npz has template arrays, not crops)
    np.savez(os.path.join(glyphs.cache_dir(None), "bank.npz"),
             templates=np.zeros((1, 4), np.float32))
    assert glyphs.available_sets(None) == ["s5"]


def test_load_cache_missing_raises(env):
    with pytest.raises(FileNotFoundError):
        glyphs.load_cache(None, "nope")


def test_build_reports_no_train_sets(env):
    make_baseline_set(env, "s6", {"p1.png": CHUNKY})
    # a set whose only entry is still pending has nothing to learn from
    src = env / "pending-src"
    src.mkdir()
    synth_display(src / "p1.png", CHUNKY)
    (src / "report.json").write_text(json.dumps({
        "frames": [{"tag": "p", "frameId": 1, "frame": CHUNKY,
                    "photo": "p1.png", "read": CHUNKY}]}), encoding="utf-8")
    curate.create_set(str(src), name="s7")  # stays pending

    result = glyphs.build()
    assert result["cells"] == 12
    assert "s6" in result["sets"]
    assert result["no_train_set"] == ["s7"]
    assert glyphs.available_sets(None) == ["s6"]
