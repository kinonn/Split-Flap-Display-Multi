"""No-ML vision scoring for split-flap alignment.

Key insight: a misaligned flap shows a *full-width* dark seam (the flap
edge shadow), while glyph strokes only ever cover part of the width. So
for each row we measure the fraction of dark pixels: seam rows are dark
across the whole module, glyph rows are not. Darkness (not gradient) is
used because a glyph overlapping the seam kills gradient contrast
(dark-on-dark) while the row stays fully dark. The threshold floats with
the crop median so exposure shifts do not matter.

Glyph *identity* is not OCR'd: the P0 index strip maps camera-X to module
index, and mechanics (forward-only drums) plus stuck detection carry
identity from there. Calibration cares about alignment, not reading text.
"""

from __future__ import annotations

import cv2
import numpy as np

# Fraction of a row's pixels that must be dark for the row to count as a
# seam candidate (seams span the module width; glyph strokes peak ~0.6).
SEAM_COL_FRAC = 0.75
# Middle band of the crop where a seam means half/double flap.
MID_BAND = (0.35, 0.65)
# Minimum separation (fraction of crop height) between two seam clusters
# to call it a double flap rather than one thick seam.
DOUBLE_SEP_FRAC = 0.10
# Mean-abs-diff below which a module is considered stuck between frames.
STUCK_DIFF = 1.5
# Canonical size for glyph-identity comparison (absorbs small crop-width
# differences from boundary refinement).
IDENTITY_SIZE = (40, 60)
# Median peer similarity below which a module is an identity outlier.
IDENTITY_THRESH = 0.85
# Absolute floor: even the best template hit below this means the crop
# matches nothing in the bank (systematic shift included). Fixture
# separation: same glyph >= 0.99, 2px-shifted 0.77, wrong glyph ~0.32.
# Confirm against field photos before tightening.
IDENTITY_ABSOLUTE_MIN = 0.6
# Glyphs this flat carry no identity information (blank crops).
IDENTITY_MIN_STD = 2.0


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def split_crops(gray: np.ndarray, n: int) -> list[np.ndarray]:
    """Split a full-display frame into n per-module crops (left to right).

    Starts from equal strips, then nudges each interior boundary to the
    nearest dark vertical gap (module gaps read dark). Falls back to the
    nominal boundary when no clear gap is found.
    """
    h, w = gray.shape[:2]
    col_brightness = np.mean(gray, axis=0)
    # Dark gaps: smooth then look for minima near nominal boundaries.
    smooth = cv2.GaussianBlur(col_brightness.reshape(1, -1), (1, 31), 0).ravel()
    bounds = [0]
    search = int(w / n * 0.06) + 2
    for i in range(1, n):
        nominal = int(round(w * i / n))
        lo, hi = max(0, nominal - search), min(w, nominal + search + 1)
        window = smooth[lo:hi]
        # Accept the minimum only if clearly darker than the strip average.
        strip_avg = float(np.mean(smooth[max(0, nominal - w // n) : min(w, nominal + w // n)]))
        best = lo + int(np.argmin(window))
        bounds.append(best if float(smooth[best]) < 0.92 * strip_avg else nominal)
    bounds.append(w)
    return [gray[:, bounds[i] : bounds[i + 1]] for i in range(n)]


def seam_profile(crop: np.ndarray) -> np.ndarray:
    """Per-row fraction of dark pixels (0..1), thresholded at half the
    crop median so the signal survives exposure shifts."""
    gray = to_gray(crop).astype(np.float32)
    if gray.size == 0:
        return np.zeros(0, dtype=np.float32)
    dark_thr = max(12.0, 0.5 * float(np.median(gray)))
    return np.mean(gray < dark_thr, axis=1).astype(np.float32)


def _clusters(rows: np.ndarray, min_gap: int = 2) -> list[tuple[int, int]]:
    """Group sorted row indices into (start, end) clusters."""
    if len(rows) == 0:
        return []
    clusters = []
    start = prev = int(rows[0])
    for r in (int(r) for r in rows[1:]):
        if r - prev > min_gap:
            clusters.append((start, prev))
            start = r
        prev = r
    clusters.append((start, prev))
    return clusters


def score_crop(crop: np.ndarray) -> dict:
    """Score one module crop. Verdict: ok | half | double."""
    gray = to_gray(crop)
    h = gray.shape[0]
    prof = seam_profile(gray)
    seam_rows = np.nonzero(prof > SEAM_COL_FRAC)[0]
    clusters = _clusters(seam_rows)
    lo, hi = int(h * MID_BAND[0]), int(h * MID_BAND[1])
    mid = [(a, b) for a, b in clusters if a <= hi and b >= lo]
    strength = float(np.max(prof)) if len(prof) else 0.0
    if len(mid) >= 2 and (mid[1][0] - mid[0][1]) / max(h, 1) >= DOUBLE_SEP_FRAC:
        verdict = "double"
    elif len(mid) >= 1:
        verdict = "half"
    else:
        verdict = "ok"
    pos = (mid[0][0] + mid[0][1]) / 2 / max(h, 1) if mid else None
    return {
        "verdict": verdict,
        "seam_clusters": [[int(a), int(b)] for a, b in clusters],
        "seam_pos": round(pos, 3) if pos is not None else None,
        "seam_strength": round(strength, 3),
    }


def stuck(prev_crop: np.ndarray, crop: np.ndarray) -> bool:
    """True when a module shows no change between two different frames."""
    a = to_gray(prev_crop).astype(np.float32)
    b = to_gray(crop).astype(np.float32)
    if a.shape != b.shape:
        return False
    return float(np.mean(np.abs(a - b))) < STUCK_DIFF


def normalize_glyph(crop: np.ndarray, size: tuple[int, int] = IDENTITY_SIZE) -> np.ndarray:
    """Resize a module crop to the canonical identity size."""
    return cv2.resize(to_gray(crop), size, interpolation=cv2.INTER_AREA)


def zncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalized cross-correlation (lighting-robust similarity).

    1.0 = identical appearance. Flat crops (no strokes, e.g. blank flaps)
    carry no identity information and always agree.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if float(a.std()) < IDENTITY_MIN_STD or float(b.std()) < IDENTITY_MIN_STD:
        return 1.0
    a -= a.mean()
    b -= b.mean()
    denom = float((a * a).sum()) * float((b * b).sum())
    if denom <= 1e-9:
        return 1.0 if float(np.abs(a - b).mean()) < 1e-9 else 0.0
    return float((a * b).sum() / np.sqrt(denom))


def consensus_outliers(crops: list[np.ndarray], thresh: float = IDENTITY_THRESH) -> list[int]:
    """Identity check for one uniform glyph shown on all modules.

    Every crop should show the SAME glyph, so any module whose appearance
    disagrees with its peers is showing the wrong (but possibly clean)
    glyph — the failure mode seam scoring cannot see. Returns the outlier
    module indices. Needs >= 3 modules to isolate blame; with fewer, the
    check abstains (returns []).
    """
    n = len(crops)
    if n < 3:
        return []
    norm = [normalize_glyph(c) for c in crops]
    outliers = []
    for i in range(n):
        sims = [zncc(norm[i], norm[j]) for j in range(n) if j != i]
        if float(np.median(sims)) < thresh:
            outliers.append(i)
    return outliers


def build_template_bank(samples: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """Build one reference template per glyph from trusted crops.

    Each crop is normalized to IDENTITY_SIZE; the template is the
    pixel-wise median (robust to one bad sample slipping in). Returns
    {glyph: template}. Callers decide trust — typically consensus winners
    (bootstrap) or a human-verified session (golden bank).
    """
    bank = {}
    for glyph, crops in samples.items():
        if not crops:
            continue
        stack = np.stack([normalize_glyph(c).astype(np.float32) for c in crops], axis=0)
        bank[glyph] = np.median(stack, axis=0).astype(np.uint8)
    return bank


def identify(crop: np.ndarray, bank: dict[str, np.ndarray]) -> list[tuple[str, float]]:
    """Rank every bank glyph by similarity to the crop (best first).

    Absolute identity: the caller compares the top hit against the
    *commanded* glyph instead of against peer modules. Flat (blank)
    templates are skipped — they match everything and prove nothing.
    """
    norm = normalize_glyph(crop)
    ranked = [(glyph, zncc(norm, template)) for glyph, template in bank.items()
              if float(template.std()) >= IDENTITY_MIN_STD]
    ranked.sort(key=lambda kv: kv[1], reverse=True)
    return ranked


def save_template_bank(bank: dict[str, np.ndarray], manifest: dict, directory: str) -> None:
    """Persist templates as PNGs plus a manifest.json (glyph -> file)."""
    import json
    import os

    os.makedirs(directory, exist_ok=True)
    files = {}
    for glyph, template in bank.items():
        filename = f"glyph_U{ord(glyph):04X}.png"
        cv2.imwrite(os.path.join(directory, filename), template)
        files[glyph] = filename
    manifest = dict(manifest)
    manifest["glyphs"] = files
    with open(os.path.join(directory, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)


def load_template_bank(directory: str) -> tuple[dict[str, np.ndarray], dict]:
    """Load a bank saved by save_template_bank. Returns (bank, manifest)."""
    import json
    import os

    with open(os.path.join(directory, "manifest.json")) as fh:
        manifest = json.load(fh)
    bank = {}
    for glyph, filename in manifest.get("glyphs", {}).items():
        img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            bank[glyph] = img
    return bank, manifest
