"""ClassifyReader tests: end-to-end local reads, gating, failure modes."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from ocr_vlm import classifier, glyphs
from ocr_vlm.classify_reader import ClassifyReader, resolve_model

from synthutil import CHUNKY, make_baseline_set


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_VLM_BASELINES", str(tmp_path / "baselines"))
    monkeypatch.setenv("OCR_VLM_DATA", str(tmp_path / "data"))
    return tmp_path


def _bank(env):
    make_baseline_set(env, "s1", {"p1.png": CHUNKY})
    payload = glyphs.extract_set("s1")
    return classifier.GlyphBank(48).fit([payload])


def _photo(env, name="p1.png"):
    path = env / "baselines" / "s1" / "images" / name
    return cv2.imread(str(path))


def test_reader_reads_curated_photo_exactly(env):
    bank = _bank(env)
    reader = ClassifyReader(backend="bank", model=bank)
    reading = reader.read(_photo(env), total=12, expected="",
                          charset=classifier.CHARSET)
    assert reading.text == CHUNKY
    assert reader.last_mode == "classify"
    assert reader.last_detected is True
    assert reader.last_display and reader.last_display["width"] > 100
    assert reader.last_blanks is not None
    assert reader.last_composed is not None      # debug montage available
    assert reader.last_raw and ":" in reader.last_raw
    classified = [m for m in reading.modules if m.source == "classifier"]
    assert len(classified) == 12
    assert all(0.0 <= m.confidence <= 1.0 for m in classified)
    assert reader.last_low_conf == 0
    assert reading.warnings == [] or all("low-confidence" not in w
                                         for w in reading.warnings)


def test_reader_flags_low_confidence_cells(env):
    bank = _bank(env)
    # The read photo is a training photo, so template confidence is a
    # perfect 1.0 — an absurd threshold guarantees every classified cell
    # lands under it and the flag path is what is under test here.
    reader = ClassifyReader(backend="bank", model=bank, min_conf=2.0)
    reading = reader.read(_photo(env), total=12)
    assert reader.last_low_conf == 12
    assert any("low-confidence" in w for w in reading.warnings)


def test_reader_without_display_reads_all_blanks(env):
    bank = _bank(env)
    reader = ClassifyReader(backend="bank", model=bank)
    uniform = np.full((220, 900, 3), 170, np.uint8)
    reading = reader.read(uniform, total=12)
    assert reader.last_no_detect is True
    assert reader.last_detected is False
    assert reading.text == " " * 12
    assert any("not detected" in w for w in reading.warnings)


def test_reader_blank_cells_skip_the_model(env):
    make_baseline_set(env, "s2", {"p1.png": "AB          "})
    payload = glyphs.extract_set("s2")
    bank = classifier.GlyphBank(48).fit([payload])
    reader = ClassifyReader(backend="bank", model=bank)
    path = env / "baselines" / "s2" / "images" / "p1.png"
    reading = reader.read(cv2.imread(str(path)), total=12)
    assert reading.text == "AB          "
    assert sum(1 for m in reading.modules if m.source == "cv") == 10
    assert any("blank cells decided by OpenCV" in w
               for w in reading.warnings)


def test_reader_requires_a_model_artifact(env):
    make_baseline_set(env, "s1", {"p1.png": CHUNKY})  # photo exists, no bank
    reader = ClassifyReader(backend="bank", data_dir=str(env / "data"))
    with pytest.raises(classifier.BankError):
        reader.read(_photo(env), total=12)
    reader = ClassifyReader(backend="cnn", data_dir=str(env / "data"))
    with pytest.raises(classifier.BankError):
        reader.read(_photo(env), total=12)


def test_resolve_model_prefers_cnn_then_bank(env):
    data = str(env / "data")
    folder = glyphs.cache_dir(data)
    os.makedirs(folder, exist_ok=True)
    bank_file = classifier.bank_path(data)
    cnn_file = os.path.join(folder, "cnn.pt")
    open(bank_file, "wb").close()
    backend, path = resolve_model(data, "auto")
    assert backend == "bank" and path == bank_file
    open(cnn_file, "wb").close()
    backend, path = resolve_model(data, "auto")
    assert backend == "cnn" and path == cnn_file
    backend, path = resolve_model(data, "bank")
    assert backend == "bank" and path == bank_file
    # an explicit path wins over the backend preference
    backend, path = resolve_model(data, "auto", explicit=bank_file)
    assert backend == "bank" and path == bank_file
    backend, path = resolve_model(data, "auto", explicit="some/model.pt")
    assert backend == "cnn" and path == "some/model.pt"


def test_reader_info_reports_provenance(env):
    bank = _bank(env)
    bank.save(classifier.bank_path(None), meta={"sets": ["s1"],
                                                "trained": "test"})
    reader = ClassifyReader(backend="bank", data_dir=str(env / "data"))
    info = reader.info()
    assert info["backend"] == "bank"
    assert info["classes"] == len(bank.classes)
    assert info["sets"] == ["s1"]
    # pipeline_info adds the artifact hash (file I/O only, no model load)
    from ocr_vlm import classify_reader
    pipe = classify_reader.pipeline_info({}, str(env / "data"))
    assert pipe["backend"] == "bank"
    assert len(pipe["sha256"]) == 64
    assert pipe["sets"] == ["s1"]
    assert pipe["classes"] == len(bank.classes)
