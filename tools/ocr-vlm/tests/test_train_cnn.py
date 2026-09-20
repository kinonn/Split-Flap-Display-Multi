"""CNN backend tests: training on synthetic caches, artifact roundtrip."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ocr_vlm import classifier, glyphs, train_cnn  # noqa: E402

from synthutil import CHUNKY, make_baseline_set  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_VLM_BASELINES", str(tmp_path / "baselines"))
    monkeypatch.setenv("OCR_VLM_DATA", str(tmp_path / "data"))
    return tmp_path


def test_glyph_data_splits_and_exclusions():
    from synthutil import cache_payload, glyph_raster

    payload = cache_payload(
        [("A", glyph_raster("A")), ("M", glyph_raster("M")),
         ("v", glyph_raster("v"))],
        photos=["s/p1.png", "s/p2.png", "s/p3.png"])
    gd = train_cnn.GlyphData([payload])
    assert gd.classes == ["A", "M"]
    assert gd.excluded == {"v": 1}
    assert len(gd.labels) == 2
    mask = gd.test_mask(0.5)
    assert mask.dtype == bool and len(mask) == 2


def test_train_all_data_and_predict(env):
    # two classes keep the tiny fit fast and deterministic: this test is
    # about the training/saving/prediction plumbing, not capacity
    content = "AMAMAMAMAMAM"
    make_baseline_set(env, "s1", {"p1.png": content, "p2.png": content,
                                  "p3.png": content})
    glyphs.build()
    result = train_cnn.train(None, epochs=60, size=32, batch=12, lr=2e-3,
                             all_data=True, verbose=False)
    assert result["classes"] == ["A", "M"]
    wrapper = train_cnn.CnnClassifier(result["model"], result["classes"],
                                      result["size"])
    caches = classifier.load_caches(None)
    correct = 0
    total = 0
    for payload in caches:
        for crop, label in zip(payload["crops"], payload["labels"]):
            pred = wrapper.predict_raster(crop)
            assert pred is not None
            assert pred[0] in result["classes"]
            assert 0.0 <= pred[1] <= 1.0
            correct += int(pred[0] == str(label))
            total += 1
    assert total == 36
    assert correct / total >= 0.9        # tiny data, trained to fit it


def test_save_load_model_roundtrip(env, tmp_path):
    make_baseline_set(env, "s1", {"p1.png": CHUNKY})
    glyphs.build()
    result = train_cnn.train(None, epochs=4, size=32, batch=16,
                             all_data=True, verbose=False)
    path = str(tmp_path / "cnn.pt")
    train_cnn.save_model(result, path, meta={"trained": "test"})
    loaded = train_cnn.CnnClassifier.load(path)
    assert loaded.classes == result["classes"]
    assert loaded.size == result["size"]
    assert loaded.meta.get("trained") == "test"
    pred = loaded.predict_raster(glyphs.load_cache(None, "s1")["crops"][0])
    assert pred is not None and pred[0] in result["classes"]


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(classifier.BankError):
        train_cnn.CnnClassifier.load(str(tmp_path / "nope.pt"))


def test_augment_preserves_shape_and_range():
    from synthutil import glyph_raster

    rng = np.random.default_rng(3)
    out = train_cnn.augment(glyph_raster("M"), rng)
    assert out.shape == (64, 64) and out.dtype == np.uint8
