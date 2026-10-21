"""Synthetic image fixtures for calib tests (no camera needed)."""

import numpy as np


def make_crop(kind="clean", w=64, h=96):
    """Fake module crop: light flap, dark glyph bars (partial width)."""
    img = np.full((h, w), 200, dtype=np.uint8)
    # Glyph strokes: E-like bars covering ~60% width + left spine.
    img[10:16, 8:46] = 40
    img[10:86, 8:14] = 40
    img[45:51, 8:42] = 40
    img[80:86, 8:46] = 40
    if kind in ("half", "double"):
        # Full-width flap seam across the middle band.
        img[h // 2 - 1 : h // 2 + 1, :] = 10
    if kind == "double":
        img[h // 2 + 12 : h // 2 + 14, :] = 10
    if kind == "half":
        # Upper glyph half sheared away above the seam (previous flap).
        img[10:16, 8:46] = 200
        img[10 : h // 2 - 1, 8:14] = 200
    return img


def make_frame(n=4, kinds=None, w=64, h=96, gap=4):
    kinds = kinds or ["clean"] * n
    parts = []
    for k in kinds:
        parts.append(make_crop(k, w, h))
        parts.append(np.full((h, gap), 20, dtype=np.uint8))  # dark module gap
    return np.concatenate(parts[:-1], axis=1)


def make_glyph(variant="E", w=64, h=96):
    """A second distinct glyph (center spine) for identity tests."""
    if variant == "E":
        return make_crop("clean", w, h)
    img = np.full((h, w), 200, dtype=np.uint8)
    img[10:86, 30:36] = 40
    img[10:16, 20:46] = 40
    img[80:86, 20:46] = 40
    return img
