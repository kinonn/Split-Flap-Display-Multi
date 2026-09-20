"""Synthetic split-flap fixtures shared by the calib-auto tests.

``synth_display`` draws a frame the real segmenter can localize and
whose per-cell glyphs are drawn with actual Hershey letter shapes, so
the glyph/classifier tests have a genuinely learnable task.
``make_baseline_set`` builds a verified golden set through the real
curation API, and ``cache_payload`` fabricates a glyph cache payload
for unit tests that do not need photos at all.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from calib_auto import golden

# 12 chars whose Hershey shapes are chunky enough that the canonical
# glyph crop's component filters (min 3 px per dimension) never drop a
# stroke — thin glyphs like 'I' are deliberately excluded.
CHUNKY = "ABCDEFGHKMNW"


def synth_display(path, labels, *, w: int = 900, h: int = 220,
                  x0: int = 100, x1: int = 700, y0: int = 40,
                  y1: int = 180, jitter: int = 0, seed: int = 0,
                  wash_modules: int = 0, wash_value: int = 120):
    """Segmentable frame whose display cells render ``labels`` as glyphs.

    Bright background + dark band (what ``find_display`` looks for);
    each label is painted with a real font so the task is separable.
    ``jitter`` (px) shifts the band and glyphs slightly per photo —
    simulates rig variance between shoots. ``wash_modules`` light-washes
    that many modules at the LEFT end of the band (flap windows go
    bright, module gaps stay dark): a lamp or window glare that pushes
    those columns above the detector's Otsu split, which is exactly what
    clips the detected run on a real rig.
    """
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 170, np.uint8)
    dy = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
    y0j, y1j = y0 + dy, min(h - 1, y1 + dy)
    img[y0j:y1j, x0:x1] = 25
    total = len(labels)
    pitch = (x1 - x0) / total
    if wash_modules > 0:
        wash_x = int(round(x0 + min(total, wash_modules) * pitch))
        img[y0j:y1j, x0:wash_x] = wash_value
    for i in range(1, total):
        xi = int(round(x0 + i * pitch))
        img[y0j:y1j, xi - 2:xi + 3] = 8
    cy = (y0j + y1j) // 2
    for i, glyph in enumerate(labels):
        if glyph == " ":
            continue
        jx = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
        cx = int(round(x0 + (i + 0.5) * pitch)) + jx
        (tw, th), _ = cv2.getTextSize(glyph, cv2.FONT_HERSHEY_SIMPLEX,
                                      1.0, 3)
        cv2.putText(img, glyph, (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3,
                    cv2.LINE_AA)
    cv2.imwrite(str(path), img)
    return str(path)


def make_baseline_set(tmp_path, name: str, contents: dict,
                      *, jitter: int = 0):
    """Verified golden set via the real curation API.

    ``contents`` maps photo filename -> 12-char content string. Photos
    are synthesized to show exactly that content. Images are kept
    byte-identical (no recompression) so tests are fast and lossless.
    """
    src = tmp_path / f"src-{name}"
    src.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, (photo, content) in enumerate(contents.items()):
        synth_display(src / photo, content, jitter=jitter, seed=i)
        frames.append({"tag": photo.split("_")[0], "frameId": i,
                       "frame": content, "photo": photo, "read": content})
    (src / "report.json").write_text(json.dumps({"frames": frames}),
                                     encoding="utf-8")
    info = golden.create_set(str(src), name=name, image_format="keep")
    for photo, content in contents.items():
        golden.update_entry(name, photo, content=content)  # verified
    return info


def glyph_raster(ch: str, *, size: int = 64, jitter: int = 0,
                 seed: int = 0) -> np.ndarray:
    """One canonical-style glyph raster (dark flap, bright strokes)."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 20, np.uint8)
    if ch != " ":
        scale = size / 64.0
        (tw, th), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX,
                                      1.1 * scale, max(2, int(3 * scale)))
        jx = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
        jy = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
        cv2.putText(img, ch, ((size - tw) // 2 + jx,
                              (size + th) // 2 + jy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1 * scale, 255,
                    max(2, int(3 * scale)), cv2.LINE_AA)
    if jitter:
        noise = rng.normal(0, 4, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(
            np.uint8)
    return img


def cache_payload(samples, *, set_name: str = "synth",
                  photos=None, positions=None, blanks=None,
                  module_count: int = 12) -> dict:
    """Minimal glyph-cache payload from (label, raster) pairs."""
    labels = [label for label, _raster in samples]
    crops = (np.stack([raster for _label, raster in samples])
             if samples else np.zeros((0, 64, 64), np.uint8))
    if photos is None:
        photos = [f"{set_name}_p{i // module_count}.png"
                  for i in range(len(samples))]
    if positions is None:
        positions = [i % module_count for i in range(len(samples))]
    if blanks is None:
        blanks = [label == " " for label in labels]
    return {
        "set": set_name,
        "module_count": module_count,
        "style": "none",
        "size": int(crops.shape[1]),
        "crops": crops,
        "labels": np.array(labels, dtype=object).astype(str),
        "blanks": np.array(blanks, dtype=bool),
        "photos": np.array(photos, dtype=object).astype(str),
        "positions": np.array(positions, dtype=np.int32),
        "summary": {"set": set_name, "cells": len(samples)},
        "meta": {"created": "", "version": 1},
    }
