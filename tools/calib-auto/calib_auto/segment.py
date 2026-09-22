"""OpenCV segmentation of split-flap display photos.

Both recognizers start here: locate the display with classical CV (no
ML, no network) and split it on the known module count, so

- per-module crops are centred on the actual modules,
- blank flaps can be decided locally (a blank flap is an all-dark
  window), which keeps isolated blank cells away from models that
  hallucinate output for them,
- the CNN gets canonical glyph rasters (``canonical_glyph``): each crop
  reduced to the glyph's own bounding box, the geometry normalization
  that makes glyphs from different rigs, exposures and band heights
  comparable,
- the module boundaries are refitted to the display's seam lattice
  (``_fit_bounds``), because the detected dark run can be clipped by a
  washed-out end of the flap band and an equal-pitch split of a clipped
  box mis-places every cell,
- crops can be normalized (contrast / inverted / binarized) before a
  VLM sees them.

Everything here works on BGR/gray numpy arrays and imports only cv2 +
numpy. There is also a CLI for validation without a model:

    python -m calib_auto.segment <photo-or-dataset-dir> [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

# A display pixel is "dark" at or below this raw gray value: the flap
# windows are near-black (measured 0..45 across the dataset), while the
# wall, table and white fixture around them sit far higher.
DARK_MAX = 45
# Column means are measured on the middle band of the frame: the top and
# bottom fractions are background (wall, table) that would dilute the
# display's signal.
BAND_FRAC = 0.15
# Column dark-run bridging and row dark-run bridging (px / frame frac):
# specular specks break columns, and the glyph band itself is *bright*,
# so the display's dark rows are split in the middle and must be joined.
COL_GAP = 30
ROW_GAP_FRAC = 0.35
# A row belongs to the display when at least this fraction of the pixels
# across the detected columns is near-black (glyph strokes are thin, so
# even a glyph row stays mostly dark).
ROW_DARK_FRAC = 0.35
# The dark and bright column classes must differ by at least this much
# (gray levels) for the split to mean "display vs background".
MIN_SEPARATION = 25.0
MIN_WIDTH_FRAC = 0.3
MIN_HEIGHT_FRAC = 0.08
# Below this confidence the caller must fall back (read as strip).
MIN_CONFIDENCE = 0.5

# Seam snapping: search this fraction of the module pitch around the
# nominal boundary and accept a darker minimum within this shift.
SEAM_SEARCH_FRAC = 0.15
SEAM_MAX_SHIFT_FRAC = 0.20
# A gap counts as a seam only if it is at most this factor of the local
# strip brightness (the same rule tools/calib's vision used).
SEAM_DARKER = 0.92
# Column smoothing (px) used by every seam decision, so the boundary
# snapper and the lattice fit look at the same profile.
SEAM_BLUR = 31

# Module-lattice fitting. The detected dark run is NOT always the whole
# display: on a rig where one end of the flap band is washed out by light
# (measured on a real run: the leftmost modules read 86..101 gray where
# the rest of the band reads 40..76, above the Otsu split) those columns
# are classified as background and the run starts *inside* the display.
# Splitting such a box into `total` equal cells shifts every cell off its
# module — the classifier then sees half-glyphs and neighbours (measured:
# 'H' read as 'T'/'5', and one '%' spanning two cells read as "'" and
# '6' on an all-blank frame).
#
# The module GAPS survive any wash (they stay dark), so the grid is
# refitted to the seam lattice: search the pitch/offset whose interior
# boundaries land on dark gaps. A lattice is only acceptable when it
# still covers the detected run, and the nominal lattice is kept whenever
# it already explains the seams — well-framed captures are untouched.
LATTICE_PITCH_MIN_FRAC = 0.5    # of the detected box's pitch
LATTICE_PITCH_MAX_FRAC = 2.5
LATTICE_PITCH_STEP = 2          # px between coarse pitch candidates
# Fine pitch step for the second pass: the coarse grid is 2px wide and a
# pitch that is off by one pixel drifts a whole module across the row.
LATTICE_PITCH_REFINE = 0.25
LATTICE_OFFSET_STEP = 2         # px between lattice offsets
# A lattice's score is how many of its interior seams the display
# *explains* (see ``_lattice_score``): a boundary counts when its snapped
# gap column is clearly darker than the two cell interiors beside it, full
# credit at this many gray levels and linear below. The score then runs
# 0..total-1, comparable across rigs, exposures and module counts.
SEAM_CREDIT_MIN = 12.0
# Cell interior is sampled over this fraction of the pitch each side of the
# cell centre and read off as a low percentile: the cell must be judged by
# the flap behind it, not by the glyph paint in front of it (a column
# through a glyph stroke is brighter than any module gap, which used to
# hand credit to any dark column at all).
SEAM_CELL_WIN_FRAC = 0.40
SEAM_CELL_QUANTILE = 20.0
# Penalty per pitch of mean seam drag, in seam credits: seams snapped far
# from where the lattice puts them mean the lattice is not really the
# gap lattice, which is what pins the reported pitch (a score saturates
# once every gap is matched, so several pitches tie without this).
LATTICE_DRAG_WEIGHT = 1.0
# A refit replaces the detected box's own grid only when it explains the
# seams by more than this many seam credits. Measured over the 421 golden
# photos: an already-aligned capture gains at most ~0.9 from any other
# lattice (sampling noise), while a box that is clipped or over-wide gains
# 6.5..9 — so this bar keeps well-framed captures byte-identical (their
# glyph caches stay valid) and still catches every real mis-fit.
LATTICE_MIN_GAIN = 1.5
# Share of interior seams the detected grid must explain before the
# search is skipped altogether: measured 0.85..1.0 on aligned captures
# and below 0.3 on the clipped/over-wide ones, so this saves the search on
# good frames without hiding a mis-fit.
LATTICE_KEEP_SUPPORT = 0.8
# Below this share of interior seams the grid does not explain the display
# (clipped detection, wrong module count, camera off-frame) and per-cell
# reads cannot be trusted: callers should escalate instead of tuning on
# them.
GRID_MIN_SUPPORT = 0.8

# Crop insets: fraction off each side of a module, and off the top and
# bottom of the display (the flap window is a band inside the housing).
CELL_INSET = 0.02
Y_INSET = 0.08

# A cell is blank when less than this fraction of pixels are ink-bright.
INK_THRESHOLD = 110.0
INK_FRACTION = 0.02
# Ink is measured on the central region only: the outermost sliver of a
# crop can hold background beyond the display's edge (the rightmost
# module of a slightly angled rig shows a bright corner) and must not
# make a blank cell look like a glyph.
INK_EDGE_TRIM = 0.10

# Montage layout: square-ish cells, index labels in a band above each.
CELL_SIZE = (128, 128)
MONTAGE_BG = 28
MONTAGE_LABEL = (220, 220, 220)
MONTAGE_COLS = 4
LABEL_BAND = 30
PAD = 8

# Canonical glyph raster: the classifier must see glyphs at a consistent
# scale and position, but the raw module window varies by photo (the
# display band height differs between rigs and shots, and the glyph sits
# at a different spot within its flap window capture to capture). Before
# caching/training, the crop is reduced to the glyph's own bounding box
# (bright ink, substantial connected components only — a specular lip
# glint is thin and low-area), padded and squared around its centre.
GLYPH_PAD = 0.08
GLYPH_MIN_AREA = 0.003
GLYPH_EDGE_TRIM = 0.10

PREPROCESS_STYLES = ("none", "contrast", "invert", "binary")
DETECTED_MODES = ("montage", "cells-detect", "strip-detect")


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _band_profile(gray: np.ndarray, y0: int, y1: int) -> np.ndarray | None:
    """Median column profile of the flap band (seams are minima).

    A median, not a mean: a column through a bright glyph still has a
    dark flap behind it, and the median tracks the flap while the mean
    drifts toward the glyph (wide glyphs like ``/`` used to wash out
    the cell-vs-gap contrast and fail the grid check on correctly
    placed cells). Same rationale as the column split in
    ``find_display``.
    """
    band = gray[max(0, y0):max(0, y1)].astype(np.float32)
    if band.size == 0:
        return None
    profile = np.median(band, axis=0).astype(np.float32)
    return cv2.GaussianBlur(profile.reshape(1, -1), (1, SEAM_BLUR),
                            0).ravel()


def _snap_seams(profile: np.ndarray, offsets: np.ndarray,
                pitch: float) -> np.ndarray:
    """Slide each seam onto the darkest column within its snap window.

    Returns the snapped *positions* (columns), like ``_snap_boundaries``
    does at crop time. A lattice must be scored the way it is used: the
    boundaries are compared at the gap they snap to, not at the nominal
    position.
    """
    n = len(profile)
    half = max(2, round(SEAM_SEARCH_FRAC * pitch))
    lo = np.clip(np.round(offsets).astype(np.int64) - half, 0, n - 1)
    win = np.clip(lo[..., None] + np.arange(2 * half + 1), 0, n - 1)
    return np.clip(lo + profile[win].argmin(axis=-1), 0, n - 1)


def _lattice_score(profile: np.ndarray, lefts: np.ndarray, pitch: float,
                   total: int, *, drag_weight: float = LATTICE_DRAG_WEIGHT
                   ) -> np.ndarray:
    """How many of a lattice's interior seams the display explains.

    One score in 0..total-1 per candidate, so it is comparable across
    rigs, exposures and module counts. A boundary is explained when its
    snapped gap column is clearly darker than the *two cells beside it* —
    full credit at ``SEAM_CREDIT_MIN`` gray levels, linear below. Both
    sides are sampled locally: the cell by a low percentile of its middle
    band (the flap, not the glyph), the gap by its snapped minimum. That
    locality is what makes the score trustworthy: a lattice off by a whole
    module has a cell on background or on its neighbour and cannot earn
    credit, where a strip-mean form could be bought with a bright
    background or a washed cell.

    ``drag_weight`` penalizes seams that had to be dragged far from where
    the lattice puts them. Ranking needs it (a score saturates once every
    gap is matched, so several pitches would tie), while the *support* a
    caller reports must not be scaled by it — pass 0.0 there.
    """
    n = len(profile)
    lefts = np.asarray(lefts, dtype=np.float64)
    if total < 2 or n < 4 or lefts.size == 0:
        return np.zeros(lefts.size, dtype=np.float64)
    win = max(1, round(SEAM_CELL_WIN_FRAC * pitch))
    centers = np.clip(np.round(
        lefts[:, None] + pitch * (np.arange(total)[None, :] + 0.5)),
        0, n - 1).astype(np.int64)
    lo = np.clip(centers - win, 0, n - 1)
    idx = np.clip(lo[..., None] + np.arange(2 * win + 1), 0, n - 1)
    cell = np.percentile(profile[idx], SEAM_CELL_QUANTILE, axis=-1)
    seams = np.clip(np.round(
        lefts[:, None] + pitch * np.arange(1, total)[None, :]),
        0, n - 1).astype(np.int64)
    snapped = _snap_seams(profile, seams, pitch)
    strip = np.minimum(cell[:, :-1], cell[:, 1:])
    credit = np.clip((strip - profile[snapped]) / SEAM_CREDIT_MIN, 0.0, 1.0)
    score = credit.sum(axis=1)
    if drag_weight:
        drag = np.abs(snapped - seams).mean(axis=1) / pitch
        score = score - drag_weight * drag
    return score


def _best_at_pitch(profile: np.ndarray, total: int, x0: float, x1: float,
                   pitch: float,
                   best: tuple[float, float, float] | None
                   ) -> tuple[float, float, float] | None:
    """Score one pitch over every offset; returns the running best.

    A candidate counts only when it and the detected dark run nest — one
    contains the other, half a module of slack each way: a clipped run
    sits inside the true lattice, an over-wide run contains it, and a
    lattice floating in the background nests with neither. Nesting is what
    keeps the pitch from collapsing onto a harmonic.
    """
    n = len(profile)
    span = pitch * total
    if span <= 0 or span > n:
        return best
    lefts = np.arange(0.0, n - span + 1.0, LATTICE_OFFSET_STEP)
    slack = 0.5 * pitch
    covers = (lefts <= x0 + slack) & (lefts + span >= x1 - slack)
    nested = (lefts >= x0 - slack) & (lefts + span <= x1 + slack)
    lefts = lefts[covers | nested]
    if lefts.size:
        scores = _lattice_score(profile, lefts, pitch, total)
        pick = int(np.argmax(scores))
        if best is None or float(scores[pick]) > best[0]:
            best = (float(scores[pick]), float(pitch), float(lefts[pick]))
    return best


def _fit_grid(profile: np.ndarray, total: int, x0: float, x1: float,
              nominal_pitch: float) -> tuple[float, float, float] | None:
    """Best (pitch, left, score) lattice covering the detected run.

    The pitch is searched around the detected box's own pitch — the box is
    either clipped by a lit end of the band (true pitch larger) or has
    swallowed background (true pitch smaller), so the two stay within a
    factor of a few — coarsely first, then refined around the winner so
    the reported pitch is good to a fraction of a pixel.
    """
    n = len(profile)
    if total < 2 or n < 8 or nominal_pitch <= 0:
        return None
    low_pitch = max(4.0, LATTICE_PITCH_MIN_FRAC * nominal_pitch)
    high_pitch = max(low_pitch + 1.0,
                     LATTICE_PITCH_MAX_FRAC * nominal_pitch)
    best: tuple[float, float, float] | None = None
    pitch = low_pitch
    while pitch <= high_pitch:
        best = _best_at_pitch(profile, total, x0, x1, pitch, best)
        pitch += LATTICE_PITCH_STEP
    if best is None:
        return None
    wide = LATTICE_PITCH_STEP + LATTICE_PITCH_REFINE
    pitch = max(low_pitch, best[1] - wide)
    while pitch <= min(high_pitch, best[1] + wide):
        best = _best_at_pitch(profile, total, x0, x1, pitch, best)
        pitch += LATTICE_PITCH_REFINE
    return best[1], best[2], best[0]


def _fit_bounds(gray: np.ndarray, bounds: list[float], y0: int, y1: int,
                total: int) -> list[float]:
    """Equal-pitch bounds -> bounds refitted to the module seams.

    The detected box's own grid is kept unless another lattice explains the
    seams clearly better (``LATTICE_MIN_GAIN``): a well-framed capture then
    keeps the exact grid it always had — so the glyph caches built from it
    stay valid — while a clipped detection, whose equal-pitch cells do not
    sit on modules at all, is replaced by the lattice that does.
    """
    if total < 2 or len(bounds) != total + 1:
        return bounds
    if not all(np.isfinite(b) for b in bounds):
        return bounds
    profile = _band_profile(gray, y0, y1)
    if profile is None:
        return bounds
    nominal_pitch = (bounds[-1] - bounds[0]) / total
    nominal = float(_lattice_score(profile, np.array([bounds[0]]),
                                   nominal_pitch, total)[0])
    if nominal >= LATTICE_KEEP_SUPPORT * (total - 1):
        return bounds
    fit = _fit_grid(profile, total, bounds[0], bounds[-1], nominal_pitch)
    if fit is None:
        return bounds
    pitch, left, score = fit
    if score <= nominal + LATTICE_MIN_GAIN:
        return bounds
    return [left + pitch * i for i in range(total + 1)]


def grid_support(image: np.ndarray,
                 boxes: list[tuple[int, int, int, int]]) -> float:
    """How much of the display a cell grid explains (0..1).

    ~1.0 means every module gap sits between two cells, i.e. the cells
    really are on the modules. A low value means the grid does not explain
    the display (clipped detection, wrong module count, camera off-frame)
    and the per-cell reads must not be trusted.
    """
    if len(boxes) < 2:
        return 1.0
    gray = to_gray(image)
    profile = _band_profile(gray, boxes[0][1], boxes[0][3])
    if profile is None:
        return 1.0
    left = float(boxes[0][0])
    total = len(boxes)
    pitch = (float(boxes[-1][2]) - left) / total
    score = float(_lattice_score(profile, np.array([left]), pitch, total,
                                 drag_weight=0.0)[0])
    return score / (total - 1)


@dataclass(frozen=True)
class DisplayBox:
    """The detected display rectangle in frame coordinates."""

    x0: float
    y0: int
    x1: float
    y1: int
    confidence: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    def as_dict(self) -> dict:
        return {"x0": round(self.x0), "y0": self.y0,
                "x1": round(self.x1), "y1": self.y1,
                "width": round(self.width), "height": self.height,
                "confidence": self.confidence}


def _longest_run(mask: np.ndarray, gap: int) -> tuple[int, int] | None:
    """(start, end) of the longest True run, tolerant to gaps <= gap px."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    best_start = best_end = start = prev = int(idx[0])
    for raw in idx[1:]:
        pos = int(raw)
        if pos - prev > gap:
            if prev - start > best_end - best_start:
                best_start, best_end = start, prev
            start = pos
        prev = pos
    if prev - start > best_end - best_start:
        best_start, best_end = start, prev
    return best_start, best_end


def _otsu_below(values: np.ndarray) -> tuple[float, np.ndarray] | tuple[None, None]:
    """Otsu-split a 1D profile: (threshold, values <= threshold).

    The comparison is inclusive: a histogram with a seam plateau right on
    the Otsu level (e.g. 22 / 38 / 170 peaks) would otherwise classify
    every pixel *at* the threshold as bright and split the display.
    """
    arr = np.ascontiguousarray(values, dtype=np.uint8).reshape(1, -1)
    if arr.size == 0 or int(arr.max()) == int(arr.min()):
        return None, None
    thr, _ = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thr), arr.ravel() <= thr


def find_display(image: np.ndarray, *,
                 min_confidence: float = MIN_CONFIDENCE) -> DisplayBox | None:
    """Locate the display's dark band, or None when nothing convincing.

    Two 1D splits, each of which the camera rig makes unambiguous:

    - columns: the mean gray of the middle band is bimodal (display vs
      wall/fixture/table), so an Otsu split yields the display's columns
      even when parts of the housing are lit;
    - rows: within those columns, a row is part of the display when a
      large fraction of it is near-black (the glyph band itself is
      bright, hence the gap-tolerant run).

    A display that is lit, cropped or off-frame fails one of the sanity
    checks and the caller falls back to reading the whole strip.
    """
    gray = to_gray(image)
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return None
    band = gray[int(BAND_FRAC * h):max(int(BAND_FRAC * h) + 1,
                                       int((1.0 - BAND_FRAC) * h)), :]
    # A low percentile (median) rather than the mean: a column that holds
    # a bright glyph still has a dark flap behind it, and the median
    # tracks the flap while the mean would drift toward the glyph.
    colmean = np.percentile(band.astype(np.float32), 50, axis=0)
    _thr, dark_cols = _otsu_below(colmean)
    if dark_cols is None:
        return None
    col_run = _longest_run(dark_cols, COL_GAP)
    if col_run is None:
        return None
    x0, x1 = col_run
    if x1 - x0 + 1 < MIN_WIDTH_FRAC * w:
        return None
    bright_mean = float(colmean[~dark_cols].mean()) if (~dark_cols).any() else 0.0
    dark_mean = float(colmean[dark_cols].mean())
    if bright_mean - dark_mean < MIN_SEPARATION:
        return None
    sub = gray[:, x0:x1 + 1].astype(np.float32)
    rowdark = (sub < DARK_MAX).mean(axis=1)
    rows = rowdark > ROW_DARK_FRAC
    # Lighting-adaptive second opinion, unioned with the absolute rule.
    # DARK_MAX was calibrated on the dark-rig photos; in brighter shots
    # the flap faces sit *above* 45 gray while the housing lip below
    # them stays under it, so the absolute rule alone clipped the
    # display to the lip (every crop pure background — caught by the
    # glyph-cache build). The Otsu split of row medians finds the dark
    # band in those shots. Either rule may over-select into dark
    # background on close-ups; that is tolerated because the run must
    # *contain* the flap band to be useful: Y_INSET trims the housing
    # and oversized crops still hold the glyph.
    rowmed = np.percentile(sub, 50, axis=1)
    _row_thr, otsu_rows = _otsu_below(rowmed)
    if otsu_rows is not None:
        rows = rows | otsu_rows
    row_run = _longest_run(rows, int(ROW_GAP_FRAC * h))
    if row_run is None:
        return None
    y0, y1 = row_run
    width, height = x1 - x0 + 1, y1 - y0 + 1
    if height < MIN_HEIGHT_FRAC * h:
        return None
    confidence = round(min(1.0, (width / w) / 0.4)
                       * min(1.0, height / (0.25 * h)), 3)
    if confidence < min_confidence:
        return None
    return DisplayBox(float(x0), y0, float(x1 + 1), y1, confidence)


def _snap_boundaries(gray: np.ndarray, bounds: list[float], y0: int,
                     y1: int) -> list[float]:
    """Nudge interior boundaries to the nearest convincing dark seam.

    The module gaps read as dark vertical lines in the flap band; the
    equal-pitch boundary is kept whenever no clearly darker column is
    found nearby, so a flat/blank display cannot pull boundaries around.
    """
    band = gray[y0:y1, :].astype(np.float32)
    if band.size == 0 or len(bounds) < 3:
        return bounds
    # Median column profile (glyph-robust, see _band_profile): a mean
    # would let bright glyph strokes near a boundary hide the seam.
    profile = np.median(band, axis=0).astype(np.float32)
    smooth = cv2.GaussianBlur(profile.reshape(1, -1), (1, SEAM_BLUR), 0).ravel()
    pitch = (bounds[-1] - bounds[0]) / (len(bounds) - 1)
    search = max(4, int(SEAM_SEARCH_FRAC * pitch))
    max_shift = SEAM_MAX_SHIFT_FRAC * pitch
    snapped = list(bounds)
    for i in range(1, len(bounds) - 1):
        nominal = bounds[i]
        lo = max(0, round(nominal - search))
        hi = min(len(smooth), round(nominal + search) + 1)
        if hi - lo < 3:
            continue
        window = smooth[lo:hi]
        pos = lo + int(np.argmin(window))
        strip_lo = max(0, round(nominal - pitch))
        strip_hi = min(len(smooth), round(nominal + pitch))
        if strip_hi <= strip_lo:
            continue
        strip_avg = float(np.mean(smooth[strip_lo:strip_hi]))
        if (abs(pos - nominal) <= max_shift
                and float(smooth[pos]) < SEAM_DARKER * strip_avg):
            snapped[i] = float(pos)
    # Clamp against the neighbours so crops can never collapse or swap.
    min_gap = max(2.0, 0.4 * pitch)
    out = [bounds[0]]
    for i in range(1, len(bounds) - 1):
        low = out[-1] + min_gap
        high = bounds[i + 1] - min_gap
        cand = snapped[i]
        out.append(min(max(cand, low), high) if high > low else bounds[i])
    out.append(bounds[-1])
    return out


def module_boxes(image: np.ndarray, display: DisplayBox, total: int, *,
                 inset: float = CELL_INSET, y_inset: float = Y_INSET,
                 snap: bool = True) -> list[tuple[int, int, int, int]]:
    """One (x0, y0, x1, y1) box per module, left to right.

    Boundaries start at equal pitch across the *detected* display, are
    refitted to the display's seam lattice (a detection clipped by a lit
    end of the band would otherwise split the modules unevenly) and are
    then optionally snapped to individual seams. The vertical range is
    inset to the flap window so housing, table and background stay out
    of the crop.
    """
    total = max(1, int(total))
    gray = to_gray(image)
    h, w = gray.shape[:2]
    x0, x1 = float(display.x0), float(display.x1)
    top = display.y0 + y_inset * display.height
    bottom = display.y1 + 1 - y_inset * display.height
    y0 = max(0, round(top))
    y1 = min(h, round(bottom))
    if y1 <= y0:
        y0, y1 = max(0, display.y0), min(h, display.y1 + 1)
    pitch = (x1 - x0) / total
    bounds = [x0 + pitch * i for i in range(total + 1)]
    if total > 1:
        bounds = _fit_bounds(gray, bounds, y0, y1, total)
    if snap and total > 1:
        bounds = _snap_boundaries(gray, bounds, y0, y1)
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(total):
        a, b = bounds[i], bounds[i + 1]
        margin = (b - a) * max(0.0, min(0.2, inset))
        xa, xb = round(a + margin), round(b - margin)
        if xb - xa < 2:
            xa, xb = round(a), round(b)
        boxes.append((max(0, xa), y0, min(w, xb), y1))
    return boxes


def crop_modules(image: np.ndarray,
                 boxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    return [image[y0:y1, x0:x1] for (x0, y0, x1, y1) in boxes]


def is_blank(crop: np.ndarray, *, ink_threshold: float = INK_THRESHOLD,
             min_ink: float = INK_FRACTION,
             edge_trim: float = INK_EDGE_TRIM) -> bool:
    """True when a module crop shows no bright glyph pixels.

    A blank split-flap window is uniformly dark; glyph strokes are
    near-white. The ink threshold floats up with the crop's own
    bright tail so a specular hotspot is not mistaken for a glyph, and
    the decision is deliberately conservative (only clearly empty cells
    are skipped — anything borderline still goes to the model).
    """
    if crop is None or getattr(crop, "size", 0) == 0:
        return True
    gray = to_gray(crop)
    h, w = gray.shape[:2]
    ty, tx = int(h * max(0.0, min(0.4, edge_trim))), int(
        w * max(0.0, min(0.4, edge_trim)))
    if h - 2 * ty >= 4 and w - 2 * tx >= 4:
        gray = gray[ty:h - ty, tx:w - tx]
    gray = gray.astype(np.float32)
    thr = max(float(ink_threshold), 0.5 * float(np.percentile(gray, 99.5)))
    return float(np.mean(gray > thr)) < min_ink


def canonical_glyph(crop, size: int, *, style: str = "none",
                    ink_threshold: float = INK_THRESHOLD,
                    pad: float = GLYPH_PAD, min_area: float = GLYPH_MIN_AREA,
                    edge_trim: float = GLYPH_EDGE_TRIM) -> np.ndarray | None:
    """Module crop -> square glyph raster, or None when there is no ink.

    The classifier's whole job is telling lookalikes apart (``0``/``O``,
    ``1``/``7``), which only works if every sample shows the glyph the
    same way. The raw module window does not: the display band height
    varies per photo and the glyph slides within the flap window, so a
    per-class pixel mean of raw crops is a blurred smudge that matches
    nothing well (measured: a clear ``7`` scored 0.60 against its own
    template). So: threshold the bright ink relative to the crop's own
    tail (the same robust rule as ``is_blank``), keep the connected
    components with real area (thin specular lip glints drop out), take
    their bounding box, pad it, square it around the centre and resize.
    Glyph geometry then means the same thing in every raster, whether
    it came from the cache build or a live read.

    Returns ``None`` when no ink component survives — the caller treats
    it as a blank cell. ``style`` is applied to the canonical raster.
    """
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    gray = to_gray(crop)
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return None
    ty = int(h * max(0.0, min(0.4, edge_trim)))
    tx = int(w * max(0.0, min(0.4, edge_trim)))
    if h - 2 * ty >= 4 and w - 2 * tx >= 4:
        inner = gray[ty:h - ty, tx:w - tx]
    else:
        inner = gray
    # Stricter than the blank test on purpose: glyph strokes are the
    # brightest ink in the cell, while the flap lip's specular glint is
    # a dimmer gray band. A high tail percentile plus per-dimension
    # minimums keeps dim, thin glints out of the bounding box — a glint
    # that joins drags the box down and shrinks the glyph, which is how
    # a clean ``5`` raster ends up half an ``S`` (measured on the ``5``
    # sweep and the bright-room rig before this rule).
    thr = max(float(ink_threshold),
              0.7 * float(np.percentile(inner, 99.9)))
    mask = (inner > thr).astype(np.uint8)
    if not mask.any():
        return None
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask,
                                                          connectivity=8)
    inner_h, inner_w = inner.shape[:2]
    min_px = max(4, int(min_area * inner.size))
    min_h = max(3, int(0.03 * inner_h))
    min_w = max(3, int(0.03 * inner_w))
    boxes = [(int(stats[c][0]), int(stats[c][1]),
              int(stats[c][2]), int(stats[c][3]))
             for c in range(1, count)
             if stats[c][4] >= min_px and stats[c][3] >= min_h
             and stats[c][2] >= min_w]
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes) + tx
    y0 = min(b[1] for b in boxes) + ty
    x1 = max(b[0] + b[2] - 1 for b in boxes) + tx
    y1 = max(b[1] + b[3] - 1 for b in boxes) + ty
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0 + 1, y1 - y0 + 1) * (1.0 + 2.0 * max(0.0, pad))
    half = side / 2.0
    left, top = round(cx - half), round(cy - half)
    right, bottom = left + round(side), top + round(side)
    gx0, gy0 = max(0, left), max(0, top)
    gx1, gy1 = min(w, right), min(h, bottom)
    if gx1 - gx0 < 2 or gy1 - gy0 < 2:
        return None
    patch = gray[gy0:gy1, gx0:gx1]
    patch = cv2.copyMakeBorder(patch, max(0, gy0 - top), max(0, bottom - gy1),
                               max(0, gx0 - left), max(0, right - gx1),
                               cv2.BORDER_REPLICATE)
    glyph = cv2.resize(patch, (int(size), int(size)),
                       interpolation=cv2.INTER_AREA)
    if style != "none":
        glyph = apply_style(glyph, style)
    return glyph


def apply_style(image: np.ndarray, style: str = "none") -> np.ndarray:
    """Style transform at native size (no resize)."""
    gray = to_gray(image)
    if style == "binary":
        _, bw = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return 255 - bw  # black glyph on white, document-like
    if style == "invert":
        return 255 - gray
    if style == "contrast":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        return clahe.apply(gray)
    return gray


def normalize(crop: np.ndarray, style: str = "none",
              size: tuple[int, int] = CELL_SIZE) -> np.ndarray:
    """One module crop -> styled, resized BGR cell image."""
    if crop is None or getattr(crop, "size", 0) == 0:
        return np.zeros((int(size[1]), int(size[0]), 3), np.uint8)
    out = apply_style(crop, style)
    out = cv2.resize(out, (int(size[0]), int(size[1])),
                     interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def montage(crops: list[np.ndarray], total: int | None = None, *,
            style: str = "none", cell: tuple[int, int] = CELL_SIZE,
            cols: int | None = None, labels: bool = True) -> np.ndarray:
    """Contact sheet of per-module crops with module index labels.

    The labels sit in their own band above each cell (not on the flap),
    so a model can keep positions straight without glyph content being
    overprinted.
    """
    total = int(total if total is not None else len(crops))
    if total <= 0:
        return np.zeros((10, 10, 3), np.uint8)
    cols = int(cols) if cols else min(MONTAGE_COLS, total)
    cols = max(1, min(cols, total))
    rows = (total + cols - 1) // cols
    cell_w, cell_h = int(cell[0]), int(cell[1])
    band = LABEL_BAND if labels else 0
    slot_w, slot_h = cell_w + PAD, cell_h + band + PAD
    canvas = np.full((rows * slot_h + PAD, cols * slot_w + PAD, 3),
                     MONTAGE_BG, np.uint8)
    for i in range(total):
        r, c = divmod(i, cols)
        x, y = PAD + c * slot_w, PAD + r * slot_h
        if labels:
            text = str(i)
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                         0.5, 1)
            cv2.putText(canvas, text, (x + (cell_w - tw) // 2,
                                       y + band - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, MONTAGE_LABEL, 1,
                        cv2.LINE_AA)
        crop = crops[i] if i < len(crops) else None
        canvas[y + band:y + band + cell_h, x:x + cell_w] = normalize(
            crop, style, (cell_w, cell_h))
    return canvas


def segment_image(image: np.ndarray, total: int, *, mode: str = "montage",
                  style: str = "none") -> tuple[np.ndarray | None, dict]:
    """Preview/CLI helper: one photo -> (visual, metadata).

    ``visual`` is the montage of normalized crops (montage/cells) or the
    styled display strip; ``None`` when the display was not detected.
    ``metadata`` carries detection, boxes and blank flags.
    """
    display = find_display(image)
    if display is None:
        return None, {"detected": False}
    total = max(1, int(total))
    boxes = module_boxes(image, display, total)
    crops = crop_modules(image, boxes)
    blanks = [is_blank(c) for c in crops]
    if mode == "strip":
        h, w = image.shape[:2]
        x0, y0 = max(0, int(display.x0)), max(0, display.y0)
        x1, y1 = min(w, int(display.x1)), min(h, display.y1 + 1)
        visual = cv2.cvtColor(apply_style(image[y0:y1, x0:x1], style),
                              cv2.COLOR_GRAY2BGR)
    else:
        visual = montage(crops, total, style=style)
    meta = {"detected": True, "display": display.as_dict(),
            "boxes": [[int(v) for v in box] for box in boxes],
            "blanks": blanks, "n_blank": int(sum(blanks))}
    return visual, meta


def _cli_stats(directory: str, total: int, style: str, mode: str,
               out: str | None) -> int:
    """Detection stats over a dataset dir (labels/reads jsonl or glob)."""
    names: list[str] = []
    manifest = None
    for candidate in ("labels.jsonl", "baseline.jsonl", "reads.jsonl"):
        if os.path.isfile(os.path.join(directory, candidate)):
            manifest = os.path.join(directory, candidate)
            break
    if manifest:
        with open(manifest, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    names.append(str(json.loads(line).get("photo", "")))
                except ValueError:
                    continue
    if not names:
        image_dir = directory
        if os.path.isdir(os.path.join(directory, "images")):
            image_dir = os.path.join(directory, "images")
        names = sorted(n for n in os.listdir(image_dir)
                       if n.lower().endswith((".png", ".jpg", ".jpeg",
                                              ".webp", ".bmp")))
    total = max(1, total)
    widths: list[float] = []
    confidences: list[float] = []
    blanks = 0
    detected = 0
    checked = 0
    for name in names:
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            path = os.path.join(directory, "images", name)
        image = cv2.imread(path)
        if image is None:
            continue
        checked += 1
        display = find_display(image)
        if display is None:
            continue
        detected += 1
        widths.append(display.width)
        confidences.append(display.confidence)
        boxes = module_boxes(image, display, total)
        blanks += sum(1 for c in crop_modules(image, boxes) if is_blank(c))
        if out:
            visual, _ = segment_image(image, total, mode=mode, style=style)
            if visual is not None:
                os.makedirs(out, exist_ok=True)
                cv2.imwrite(os.path.join(
                    out, os.path.splitext(name)[0] + ".seg.png"), visual)
    stats = {
        "photos": checked,
        "detected": detected,
        "fallback": checked - detected,
        "detect_rate": round(detected / checked, 3) if checked else 0.0,
        "width_mean": round(float(np.mean(widths)), 1) if widths else None,
        "width_std": round(float(np.std(widths)), 1) if widths else None,
        "width_min": int(min(widths)) if widths else None,
        "width_max": int(max(widths)) if widths else None,
        "confidence_mean": round(float(np.mean(confidences)), 3)
        if confidences else None,
        "blanks": blanks,
    }
    print(json.dumps(stats, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Segment split-flap display photos (OpenCV only).")
    parser.add_argument("path", help="photo file or dataset directory")
    parser.add_argument("--total", type=int, default=12,
                        help="module count (default 12)")
    parser.add_argument("--mode", default="montage",
                        choices=("montage", "cells", "strip"))
    parser.add_argument("--style", default="none", choices=PREPROCESS_STYLES)
    parser.add_argument("--out", default=None,
                        help="output PNG path/dir (written when set)")
    parser.add_argument("--json", action="store_true",
                        help="print metadata as JSON")
    args = parser.parse_args(argv)

    if os.path.isdir(args.path):
        return _cli_stats(args.path, args.total, args.style, args.mode,
                          args.out)

    image = cv2.imread(args.path)
    if image is None:
        print(f"cannot read image: {args.path}", file=sys.stderr)
        return 2
    visual, meta = segment_image(image, args.total, mode=args.mode,
                                 style=args.style)
    if visual is not None and args.out:
        cv2.imwrite(args.out, visual)
    if args.json:
        print(json.dumps(meta, indent=2))
    elif meta.get("detected"):
        box = meta["display"]
        print(f"display x {box['x0']}..{box['x1']} y {box['y0']}..{box['y1']} "
              f"({box['width']}x{box['height']}, conf {box['confidence']})")
        print(f"blanks: {meta['n_blank']}/{args.total} "
              f"{meta['blanks']}")
        print("boxes: " + ", ".join(f"{b[0]}-{b[2]}" for b in meta["boxes"]))
    else:
        print("display not detected (would fall back to strip)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
