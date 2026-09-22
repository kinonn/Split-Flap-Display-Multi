"""Segmentation tests: synthetic split-flap frames, no models involved."""

from __future__ import annotations

import json

import cv2
import numpy as np

from ocr_vlm.segment import (CELL_SIZE, LABEL_BAND, MONTAGE_COLS, PAD,
                             DisplayBox, apply_style, crop_modules,
                             find_display, is_blank, main, module_boxes,
                             montage, normalize, segment_image)


def _display_image(w: int = 900, h: int = 220, x0: int = 100, x1: int = 700,
                   y0: int = 40, y1: int = 180, total: int = 12,
                   glyphs: str = "", seam_shift: int = 0,
                   bg: int = 170) -> np.ndarray:
    """A bright background with a dark display band and darker seams."""
    img = np.full((h, w, 3), bg, np.uint8)
    img[y0:y1, x0:x1] = 25
    pitch = (x1 - x0) / total
    for i in range(1, total):
        xi = int(round(x0 + i * pitch + seam_shift))
        img[y0:y1, xi - 2:xi + 3] = 8
    if glyphs:
        cy = (y0 + y1) // 2
        for i, ch in enumerate(glyphs):
            if ch in (" ", ""):
                continue
            a = int(round(x0 + i * pitch)) + 8
            b = int(round(x0 + (i + 1) * pitch)) - 8
            img[cy - 25:cy + 25, a:b] = 255
    return img


# -- find_display -------------------------------------------------------------

def test_find_display_locates_the_dark_band():
    img = _display_image()
    box = find_display(img)
    assert box is not None
    assert abs(box.x0 - 100) <= 3 and abs(box.x1 - 700) <= 3
    assert abs(box.y0 - 40) <= 3 and abs(box.y1 - 180) <= 3
    assert box.confidence > 0.8
    assert box.as_dict()["width"] > 500


def test_find_display_none_on_uniform_background():
    assert find_display(np.full((220, 900, 3), 170, np.uint8)) is None


def test_find_display_none_on_tiny_dark_blob():
    img = np.full((220, 900, 3), 170, np.uint8)
    img[80:140, 40:180] = 10  # too narrow to cover MIN_RUN_FRAC
    assert find_display(img) is None


def test_find_display_prefers_the_wider_dark_region():
    img = _display_image()
    img[0:220, 0:60] = 10  # a dark object at the left frame edge
    box = find_display(img)
    assert box is not None and abs(box.x0 - 100) <= 3


# -- module_boxes -------------------------------------------------------------

def test_module_boxes_split_equal_pitch_and_inset_vertically():
    img = _display_image()
    box = find_display(img)
    assert box is not None
    boxes = module_boxes(img, box, 12)
    assert len(boxes) == 12
    widths = [b[2] - b[0] for b in boxes]
    assert max(widths) - 48 <= 2                 # ~50 px pitch, 2 % inset
    for i in range(1, 12):
        assert boxes[i][0] >= boxes[i - 1][2] - 1
        assert boxes[i][0] > boxes[i - 1][0]
    assert boxes[0][0] >= int(box.x0)
    assert boxes[-1][2] <= int(box.x1) + 1
    assert boxes[0][1] > box.y0 and boxes[0][3] < box.y1 + 1


def test_module_boxes_snap_to_offset_seam():
    img = _display_image(seam_shift=6)
    box = find_display(img)
    assert box is not None
    nominal = module_boxes(img, box, 12, snap=False)
    snapped = module_boxes(img, box, 12, snap=True)
    shift = abs(snapped[0][2] - nominal[0][2]) + abs(
        snapped[1][0] - nominal[1][0])
    assert shift >= 4, "seam 6 px off nominal should pull a boundary"
    # The snap stays near the seam, not the nominal line.
    assert abs((snapped[0][2] + snapped[1][0]) / 2 - (100 + 50 + 6)) <= 3


def test_module_boxes_do_not_collapse_on_flat_display():
    img = _display_image()
    box = find_display(img)
    assert box is not None
    boxes = module_boxes(img, box, 12, snap=True)
    for x0, _y0, x1, _y1 in boxes:
        assert x1 - x0 >= 20


def test_crop_modules_returns_image_slices():
    img = _display_image(glyphs="H H H H H H ")
    box = find_display(img)
    assert box is not None
    boxes = module_boxes(img, box, 12)
    crops = crop_modules(img, boxes)
    assert len(crops) == 12
    assert crops[0].shape[0] == boxes[0][3] - boxes[0][1]
    assert int(crops[0].max()) == 255            # glyph block crop
    assert int(crops[1].max()) < 100             # blank crop


# -- blank detection ----------------------------------------------------------

def test_is_blank_on_dark_vs_glyph_crop():
    dark = np.full((80, 60, 3), 25, np.uint8)
    assert is_blank(dark) is True
    assert is_blank(np.zeros((0, 0, 3), np.uint8)) is True
    glyph = dark.copy()
    glyph[20:60, 15:45] = 255
    assert is_blank(glyph) is False


def test_is_blank_tolerates_specular_hotspot():
    dark = np.full((80, 60, 3), 25, np.uint8)
    dark[0:7, 0:7] = 200                          # ~1 % of the crop
    assert is_blank(dark) is True


# -- normalize / styles -------------------------------------------------------

def test_apply_style_binary_is_black_on_white():
    crop = np.full((40, 40), 25, np.uint8)
    crop[10:30, 10:30] = 255
    out = apply_style(crop, "binary")
    assert out.shape == crop.shape
    assert out[0, 0] > 127 and out[20, 20] < 127


def test_normalize_returns_fixed_size_bgr():
    crop = np.full((80, 60, 3), 25, np.uint8)
    out = normalize(crop, "none", (64, 48))
    assert out.shape == (48, 64, 3)
    assert out.ndim == 3


def test_normalize_empty_crop_is_a_blank_cell():
    out = normalize(np.zeros((0, 0, 3), np.uint8), "binary", (32, 32))
    assert out.shape == (32, 32, 3) and int(out.min()) == 0


# -- montage ------------------------------------------------------------------

def test_montage_layout_and_labels():
    crops = [np.full((80, 60, 3), 25 + i, np.uint8) for i in range(12)]
    sheet = montage(crops, 12, cols=4)
    assert sheet.shape == (3 * (CELL_SIZE[1] + LABEL_BAND + PAD) + PAD,
                           4 * (CELL_SIZE[0] + PAD) + PAD, 3)
    labels = np.argwhere(np.all(sheet == 220, axis=2))
    assert len(labels) > 0                        # index digits drawn


def test_montage_default_columns_capped_at_four():
    crops = [np.full((40, 40, 3), 25, np.uint8)] * 6
    sheet = montage(crops, 6)
    expected_w = 4 * (CELL_SIZE[0] + PAD) + PAD
    assert sheet.shape[1] == expected_w
    assert MONTAGE_COLS == 4


def test_montage_handles_missing_crops():
    sheet = montage([], 3, labels=False)
    assert sheet.shape[0] > 0 and sheet.shape[1] > 0


# -- segment_image ------------------------------------------------------------

def test_segment_image_montage_and_metadata():
    img = _display_image(glyphs="AB          ")
    visual, meta = segment_image(img, 12)
    assert meta["detected"] is True
    assert len(meta["boxes"]) == 12
    assert meta["blanks"][0] is False and meta["blanks"][3] is True
    assert visual is not None and visual.ndim == 3


def test_segment_image_strip_mode_keeps_display_extent():
    img = _display_image()
    visual, meta = segment_image(img, 12, mode="strip")
    assert meta["detected"] is True
    assert visual is not None
    assert visual.shape[1] == meta["display"]["width"]
    assert visual.shape[0] == meta["display"]["height"]


def test_segment_image_without_display():
    visual, meta = segment_image(np.full((220, 900, 3), 170, np.uint8), 12)
    assert visual is None and meta == {"detected": False}


# -- CLI ----------------------------------------------------------------------

def test_cli_single_photo_json(tmp_path, capsys):
    path = tmp_path / "shot.png"
    cv2.imwrite(str(path), _display_image(glyphs="H" * 12))
    assert main([str(path), "--json"]) == 0
    meta = json.loads(capsys.readouterr().out)
    assert meta["detected"] is True and meta["n_blank"] == 0


def test_cli_dataset_dir_stats(tmp_path, capsys):
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"sw_{i}_f{i}.png"),
                    _display_image(glyphs="H" * 12))
    cv2.imwrite(str(tmp_path / "blank.png"),
                np.full((220, 900, 3), 170, np.uint8))
    rows = [{"photo": f"sw_{i}_f{i}.png", "want": "H" * 12, "saw": "H" * 12}
            for i in range(3)]
    rows.append({"photo": "blank.png", "want": " " * 12, "saw": " " * 12})
    (tmp_path / "reads.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert main([str(tmp_path)]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["photos"] == 4 and stats["detected"] == 3
    assert stats["fallback"] == 1
    assert stats["width_mean"] is not None


def test_display_box_roundtrip():
    box = DisplayBox(10.4, 20, 300.6, 200, 0.9)
    assert box.width == 300.6 - 10.4
    assert box.height == 181
    assert box.as_dict()["confidence"] == 0.9
