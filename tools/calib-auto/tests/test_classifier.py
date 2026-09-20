"""Template bank tests: fit/predict, charset filter, gating, evaluation."""

from __future__ import annotations

import os

import numpy as np
import pytest

from calib_auto import classifier, glyphs, segment
from calib_auto.classifier import (GlyphBank, classify_cell, classify_raster)

from synthutil import (CHUNKY, cache_payload, glyph_raster,
                       make_baseline_set)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_AUTO_DATA", str(tmp_path))
    return tmp_path


def _canon(ch):
    """A glyph raster on the canonical geometry the cache stores."""
    raster = segment.canonical_glyph(glyph_raster(ch), glyphs.GLYPH_SIZE)
    assert raster is not None
    return raster


def _samples(chars, per=2):
    return [(ch, _canon(ch)) for ch in chars for _ in range(per)]


def test_fit_predict_roundtrip():
    bank = GlyphBank(48).fit([cache_payload(_samples("AMW7"))])
    for ch in "AMW7":
        char, conf, margin = bank.predict_raster(_canon(ch))
        assert char == ch
        assert conf > 0.9
        assert margin >= 0.0
    assert bank.counts == {"7": 2, "A": 2, "M": 2, "W": 2}


def test_predict_live_crop_canonicalizes():
    bank = GlyphBank(48).fit([cache_payload(_samples("AM"))])
    char, conf, _margin = bank.predict(glyph_raster("A"))
    assert char == "A"
    assert conf > 0.5


def test_fit_excludes_labels_outside_charset():
    payload = cache_payload(_samples("AM") + [("v", glyph_raster("v"))])
    bank = GlyphBank(48).fit([payload])
    assert "v" not in bank.classes
    assert bank.skipped_labels == {"v": 1}


def test_fit_skips_zero_rasters():
    zeros = np.zeros((64, 64), np.uint8)
    payload = cache_payload([("A", glyph_raster("A")), ("A", zeros),
                             ("M", glyph_raster("M"))])
    bank = GlyphBank(48).fit([payload])
    assert bank.counts == {"A": 1, "M": 1}


def test_fit_without_usable_crops_raises():
    with pytest.raises(classifier.BankError):
        GlyphBank(48).fit([cache_payload([])])


def test_save_load_roundtrip(tmp_path):
    bank = GlyphBank(48).fit([cache_payload(_samples("AM"))])
    path = str(tmp_path / "bank.npz")
    bank.save(path, meta={"sets": ["s1"]})
    loaded = GlyphBank.load(path)
    assert loaded.classes == bank.classes
    assert loaded.counts == bank.counts
    assert loaded.size == bank.size
    assert loaded.meta.get("sets") == ["s1"]
    a = bank.predict_raster(_canon("M"))
    b = loaded.predict_raster(_canon("M"))
    assert a[0] == b[0] == "M"
    assert os.path.isfile(str(tmp_path / "bank.json"))


def test_load_missing_and_corrupt(tmp_path):
    with pytest.raises(classifier.BankError):
        GlyphBank.load(str(tmp_path / "nope.npz"))
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not an npz")
    with pytest.raises(classifier.BankError):
        GlyphBank.load(str(bad))


def test_classify_raster_gating_policy():
    bank = GlyphBank(48).fit([cache_payload(_samples("AM"))])
    # Opt-in blank gate: a bright compact blob survives even when the ink
    # test called the cell blank (this is the period/apostrophe rescue)
    dot = np.full((64, 64), 20, np.uint8)
    dot[26:38, 26:38] = 255
    char, conf, margin, source = classify_raster(
        bank, dot, blank=True, blank_gate=True)
    assert source == "classifier"
    # dim texture stays blank when the ink test also says blank
    dim = np.full((64, 64), 30, np.uint8)
    dim[30:34, 20:50] = 100
    char, conf, margin, source = classify_raster(
        bank, dim, blank=True, blank_gate=True)
    assert (char, source) == (" ", "cv")
    # a zero raster is always a blank, whatever the ink test said
    zeros = np.zeros((64, 64), np.uint8)
    assert classify_raster(bank, zeros, blank=False)[0] == " "
    assert classify_raster(
        bank, zeros, blank=False, blank_gate=True)[0] == " "
    # and a bright glyph that the ink test also flagged is classified
    char, conf, margin, source = classify_raster(
        bank, _canon("M"), blank=False)
    assert (char, source) == ("M", "classifier")


def test_classify_raster_gate_off_by_default():
    """Default: the ink-test flag never short-circuits to OpenCV."""
    bank = GlyphBank(48).fit([cache_payload(_samples("AM"))])
    dim = np.full((64, 64), 30, np.uint8)
    dim[30:34, 20:50] = 100
    # same dim cell the gate would call blank goes to the model instead
    char, _conf, _margin, source = classify_raster(bank, dim, blank=True)
    assert source == "classifier"
    # zero rasters are classifier blanks, not cv, when the gate is off
    char, _conf, _margin, source = classify_raster(
        bank, np.zeros((64, 64), np.uint8), blank=True)
    assert (char, source) == (" ", "classifier")


def test_classify_cell_matches_raster_path():
    bank = GlyphBank(48).fit([cache_payload(_samples("AM"))])
    char, _conf, _margin, source = classify_cell(
        bank, glyph_raster("A"), blank=False)
    assert (char, source) == ("A", "classifier")


def test_test_mask_is_deterministic_and_photo_based():
    keys = [f"set/photo_{i}.png" for i in range(40)]
    mask = classifier._test_mask(keys, "photo", None, ["set"] * 40, 0.25)
    again = classifier._test_mask(keys, "photo", None, ["set"] * 40, 0.25)
    assert mask.tolist() == again.tolist()
    assert 0 < int(mask.sum()) < 40
    with pytest.raises(classifier.BankError):
        classifier._test_mask(keys, "set", None, ["set"] * 40, 0.25)


def test_load_caches_subset_and_missing(env):
    make_baseline_set(env, "s1", {"p1.png": CHUNKY})
    make_baseline_set(env, "s2", {"p1.png": CHUNKY})
    glyphs.build()
    caches = classifier.load_caches(["s1"])
    assert [c["set"] for c in caches] == ["s1"]
    with pytest.raises(classifier.BankError, match="no glyph cache"):
        classifier.load_caches(["nope"])


def test_evaluate_set_holdout_on_synthetic_sets(env):
    make_baseline_set(env, "s1", {"p1.png": CHUNKY, "p2.png": CHUNKY})
    make_baseline_set(env, "s2", {"p1.png": CHUNKY, "p2.png": CHUNKY})
    glyphs.build()
    metrics = classifier.evaluate(split="set", holdout="s2")
    assert metrics["train_sets"] == ["s1"]
    assert metrics["test_sets"] == ["s2"]
    assert metrics["test_cells"] == 24
    assert metrics["sim_acc"] >= 0.9
    assert metrics["rows"] == 2
    assert 0.0 <= metrics["row_exact_rate"] <= 1.0
    assert "per_class" in metrics and "confusions" in metrics
