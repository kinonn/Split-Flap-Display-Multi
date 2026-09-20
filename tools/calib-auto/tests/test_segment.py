"""Module-grid fitting: cells must sit on modules, and say so when they don't.

The display detector returns the longest dark run, which a lit end of the
flap band clips (real rig: the leftmost modules read 86..101 gray where
the rest of the band reads 40..76, so those columns fall on the
background side of the Otsu split). Splitting such a run into `total`
equal cells puts every cell off its module, and the classifier then reads
half-glyphs and neighbours as if they were characters — the failure that
produced a garbage `cal-001`. These tests pin the refit, the support
measure and the reader's honesty about a grid it cannot trust.
"""

from __future__ import annotations

import cv2
import numpy as np

from calib_auto import cnn_reader, glyphs, segment
from calib_auto.classifier import GlyphBank

from synthutil import CHUNKY, cache_payload, synth_display

TOTAL = 12
LABELS = CHUNKY  # 12 chunky glyphs, one per module
X0, X1 = 100, 700
PITCH = (X1 - X0) / TOTAL


def _bank(tmp_path, chars) -> GlyphBank:
    """A bank trained on the fixture's own glyph rendering.

    One module per frame, canonicalized exactly like a live read — the
    same route a real glyph cache takes from curated photos of the rig.
    These tests are about the module grid, so the classifier should not
    also have to transfer between two renderings.
    """
    samples = []
    for i, ch in enumerate(chars):
        path = tmp_path / f"cell{i}_{ord(ch)}.png"
        synth_display(path, ch, w=300, h=200, x0=100, x1=200, y0=40, y1=160)
        img = cv2.imread(str(path))
        display = segment.find_display(img)
        assert display is not None
        crop = segment.crop_modules(
            img, segment.module_boxes(img, display, 1))[0]
        raster = segment.canonical_glyph(crop, glyphs.GLYPH_SIZE)
        assert raster is not None
        samples.append((ch, raster))
    return GlyphBank(48).fit([cache_payload(samples)])


def _true_edges() -> list[float]:
    return [X0 + PITCH * i for i in range(TOTAL + 1)]


def _load(path):
    img = cv2.imread(str(path))
    display = segment.find_display(img)
    assert display is not None
    return img, display, segment.module_boxes(img, display, TOTAL)


def _assert_on_modules(boxes, tol: float = 4.0):
    for (bx0, _by0, bx1, _by1), lo, hi in zip(boxes, _true_edges(),
                                              _true_edges()[1:]):
        assert abs(bx0 - lo) <= tol, f"cell left {bx0} vs module {lo}"
        assert abs(bx1 - hi) <= tol, f"cell right {bx1} vs module {hi}"


def test_cells_sit_on_modules_even_when_the_run_is_clipped(tmp_path):
    path = synth_display(tmp_path / "washed.png", LABELS, wash_modules=3)
    img, display, boxes = _load(path)
    # the wash clips the detected run: it starts inside the display
    assert display.x0 > X0 + 1.5 * PITCH
    _assert_on_modules(boxes)
    assert segment.grid_support(img, boxes) == 1.0


def test_clean_frame_keeps_its_grid(tmp_path):
    path = synth_display(tmp_path / "clean.png", LABELS)
    img, _display, boxes = _load(path)
    _assert_on_modules(boxes)
    assert segment.grid_support(img, boxes) == 1.0


def test_grid_support_drops_when_cells_are_off_module(tmp_path):
    img, _display, boxes = _load(synth_display(tmp_path / "clean.png",
                                               LABELS))
    shifted = [(b[0] + int(PITCH / 2), b[1], b[2] + int(PITCH / 2), b[3])
               for b in boxes]
    assert segment.grid_support(img, shifted) < segment.GRID_MIN_SUPPORT


def test_reader_reads_a_washed_frame(tmp_path):
    """End to end: the wash used to turn these rows into noise."""
    img = cv2.imread(str(synth_display(tmp_path / "washed.png", LABELS,
                                       wash_modules=3)))
    reader = cnn_reader.CnnReader(backend="bank",
                                  model=_bank(tmp_path, LABELS))
    reading = reader.read(img, TOTAL, expected=LABELS)
    assert reading.text == LABELS
    assert not reading.unreliable
    assert reading.unreliable_note == ""
    assert all(m.condition == "clean" for m in reading.modules)


def test_reader_flags_a_grid_it_cannot_verify(tmp_path):
    """No visible module gaps: the cells cannot be shown to be on modules."""
    img = np.full((220, 900, 3), 170, np.uint8)
    img[40:180, X0:X1] = 25  # uniform band, no seams
    reader = cnn_reader.CnnReader(backend="bank", model=_bank(tmp_path, "A"))
    reading = reader.read(img, TOTAL, expected="A" * TOTAL)
    assert reading.unreliable
    assert "does not fit" in reading.unreliable_note
    assert any("does not fit" in w for w in reading.warnings)
    # a cell that is not on its module is not a reading: no confidence,
    # no "clean" condition, so the loop cannot tune on it
    assert all(m.confidence == 0.0 for m in reading.modules)
    assert all(m.condition == "unreadable" for m in reading.modules)


def test_reader_flags_an_undetected_display(tmp_path):
    img = np.full((220, 900, 3), 170, np.uint8)  # nothing dark to locate
    reader = cnn_reader.CnnReader(backend="bank", model=_bank(tmp_path, "A"))
    reading = reader.read(img, TOTAL, expected="A" * TOTAL)
    assert reading.unreliable
    assert "not detected" in reading.unreliable_note
    assert all(m.char == "?" for m in reading.modules)
    # all '?' is what P0's index gate rejects, so the run stops there
    # instead of sweeping a display whose modules were never located
