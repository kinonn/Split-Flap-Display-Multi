"""Local glyph classifier: template bank backend + honest evaluation.

Stage B of the custom-recognizer plan. The rules of this module:

- **Verified truth only.** Training data comes from glyph caches built
  by ``glyphs.py``, which only ever reads ``verified`` baseline
  entries. Pending pre-fills never train or score anything.
- **Photo-disjoint evaluation.** Accuracy is never reported on photos
  the bank trained on: ``split="photo"`` holds out a hash-selected
  fifth of the photos, ``split="set"`` holds out a whole baseline set
  (leave-one-set-out — the model is then tested on a shoot it has
  never seen, which is what deployment looks like).
- **Two numbers, both true.** ``bank_acc`` runs the classifier on every
  cell (blanks included as their own class, when training saw them);
  ``sim_acc`` simulates the reader — OpenCV decides blanks, the bank
  classifies the rest — and reports whole-row mismatches, the unit the
  benchmark scores.

The bank itself is deliberately dependency-free: per class the mean of
unit-normalized grayscale crops (resized to ``size``), prediction by
cosine similarity, ``(char, confidence, margin)`` out. It is fast, tiny
(~a few hundred KB) and on clean curated crops it is already strong;
the CNN backend (``train_cnn.py``) exists for glare/pose robustness
and is a drop-in replacement behind the same interface.

CLI::

    python -m ocr_vlm.classifier build [--data DIR] [--size 48] [--sets ...]
    python -m ocr_vlm.classifier eval  [--split all|photo|set] [--holdout NAME]
    python -m ocr_vlm.classifier read  PHOTO [--modules 12]
"""

from __future__ import annotations

import argparse
import json
import os
import zlib
from datetime import datetime, timezone

import cv2
import numpy as np

from . import glyphs, segment

BANK_FILE = "bank.npz"
BANK_META = "bank.json"
DEFAULT_SIZE = 48
BLANK = " "
# The display's drum character set (must match server.DEFAULT_CHARSET and
# src/web). Curated labels outside it are typos/transients (one lowercase
# 'v' was found); they are excluded from training and counted in the
# bank meta so the data problem stays visible instead of becoming a
# phantom class.
CHARSET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':?!.-/$@#%"
# Blank-vs-glyph decision on the canonical raster: real glyph ink is
# bright even when tiny (the '.' dot: p99 193+) while blank-flap texture
# is dim (p99 ~133-153); 170 separates them with only ~10 dim glyph
# outliers dataset-wide. Cells that are bright go to the classifier
# even when OpenCV's ink test called them blank — that is what rescues
# the period and apostrophe, whose ink area is under the blank frac.
GLYPH_BRIGHT_FLOOR = 170.0


class BankError(RuntimeError):
    """A bank/evaluation problem the caller should surface as a 4xx."""


def bank_path(data: str | None = None) -> str:
    return os.path.join(glyphs.cache_dir(data), BANK_FILE)


def bank_meta_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".json"


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    vec -= vec.mean()
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return vec
    return vec / norm


class GlyphBank:
    """Per-class mean templates on unit-normalized grayscale crops."""

    def __init__(self, size: int = DEFAULT_SIZE):
        self.size = max(8, int(size))
        self.classes: list[str] = []
        self.templates: np.ndarray | None = None  # (n_classes, size*size)
        self.counts: dict[str, int] = {}
        self.meta: dict = {}

    # -- training --------------------------------------------------------------

    def _vec(self, raster) -> np.ndarray | None:
        """Canonical glyph raster -> unit-normalized feature vector.

        The raster is already the glyph's own bounding box at
        ``GLYPH_SIZE`` (``segment.canonical_glyph``), so training and
        live inference share this one resampling step to the bank's
        work size — the geometry normalization that makes per-class
        means meaningful happens upstream, once.
        """
        if raster is None or getattr(raster, "size", 0) == 0:
            return None
        gray = segment.to_gray(raster)
        if gray.shape[:2] != (self.size, self.size):
            gray = cv2.resize(gray, (self.size, self.size),
                              interpolation=cv2.INTER_AREA)
        return _unit(gray.reshape(-1))

    def fit(self, caches: list[dict], *,
            charset: str | None = None) -> GlyphBank:
        """Train from glyph-cache payloads (see ``glyphs.extract_set``).

        All-zero rasters (blank cells: the cache stores a zero raster
        when no ink was found) are skipped — their unit vector is the
        zero vector, which would drag every class mean it joins toward
        zero. Blank cells are gated by the reader's ink test on the
        read path, never by the bank. Labels outside ``charset`` (the
        drum's real characters) are excluded and counted — a curated
        typo must not become a phantom class.
        """
        allowed = set(charset) if charset is not None else set(CHARSET)
        groups: dict[str, list[np.ndarray]] = {}
        self.skipped_labels: dict[str, int] = {}
        for payload in caches:
            crops, labels = payload["crops"], payload["labels"]
            for crop, label in zip(crops, labels):
                label = str(label)
                if label not in allowed:
                    self.skipped_labels[label] = \
                        self.skipped_labels.get(label, 0) + 1
                    continue
                vec = self._vec(crop)
                if vec is None or not vec.any():
                    continue
                groups.setdefault(label, []).append(vec)
        if not groups:
            raise BankError("no crops to train on — build glyph caches first")
        self.classes = sorted(groups)
        templates = []
        for char in self.classes:
            mean = np.mean(np.stack(groups[char]), axis=0)
            templates.append(_unit(mean))
        self.templates = np.stack(templates).astype(np.float32)
        self.counts = {char: len(groups[char]) for char in self.classes}
        return self

    # -- inference -------------------------------------------------------------

    @property
    def trained(self) -> bool:
        return self.templates is not None and bool(self.classes)

    def predict_raster(self, raster) -> tuple[str, float, float] | None:
        """Canonical raster -> (char, confidence, margin) (cache path)."""
        if not self.trained:
            raise BankError("bank is not trained")
        vec = self._vec(raster)
        if vec is None:
            return None
        scores = self.templates @ vec
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        conf = float(scores[best])
        second = float(scores[int(order[1])]) if len(order) > 1 else 0.0
        return self.classes[best], conf, conf - second

    def predict(self, crop) -> tuple[str, float, float] | None:
        """Live module crop -> canonicalize, then classify (read path).

        None means the cell produced no usable ink (treat as blank).
        The canonicalization is exactly the one the cache build used,
        so a live crop and the same crop in training land on identical
        rasters.
        """
        raster = segment.canonical_glyph(crop, glyphs.GLYPH_SIZE)
        if raster is None:
            return None
        return self.predict_raster(raster)

    # -- persistence -----------------------------------------------------------

    def save(self, path: str, meta: dict | None = None) -> str:
        if not self.trained:
            raise BankError("nothing to save — bank is not trained")
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez_compressed(
                fh, templates=self.templates,
                classes=np.array(self.classes),
                counts=np.array([self.counts[c] for c in self.classes]),
                size=np.int32(self.size))
        os.replace(tmp, path)
        payload = {
            "created": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "backend": "bank",
            "size": self.size,
            "classes": self.classes,
            "counts": self.counts,
            "cells": int(sum(self.counts.values())),
        }
        payload.update(meta or {})
        self.meta = payload
        tmp = bank_meta_path(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, bank_meta_path(path))
        return path

    @classmethod
    def load(cls, path: str) -> GlyphBank:
        if not os.path.isfile(path):
            raise BankError(f"no classifier bank at {path}")
        try:
            with np.load(path, allow_pickle=False) as npz:
                bank = cls(size=int(npz["size"]))
                bank.templates = npz["templates"].astype(np.float32)
                bank.classes = [str(c) for c in npz["classes"]]
                counts = npz["counts"]
                bank.counts = {c: int(n)
                               for c, n in zip(bank.classes, counts)}
        except (OSError, ValueError, KeyError) as exc:
            raise BankError(f"corrupt classifier bank at {path}: {exc}") \
                from exc
        meta_file = bank_meta_path(path)
        if os.path.isfile(meta_file):
            try:
                with open(meta_file, encoding="utf-8") as fh:
                    bank.meta = json.load(fh)
            except (OSError, ValueError):
                bank.meta = {}
        return bank


def load_caches(data: str | None = None,
                sets: list[str] | None = None) -> list[dict]:
    """Load every available glyph cache (optionally a subset)."""
    names = glyphs.available_sets(data)
    if sets:
        wanted = set(sets)
        names = [n for n in names if n in wanted]
        missing = wanted - set(names)
        if missing:
            raise BankError(
                "no glyph cache for: " + ", ".join(sorted(missing))
                + " — run: python -m ocr_vlm.glyphs")
    return [glyphs.load_cache(data, name) for name in names]


def train_bank(data: str | None = None, *,
               sets: list[str] | None = None,
               size: int = DEFAULT_SIZE,
               out: str | None = None) -> GlyphBank:
    caches = load_caches(data, sets)
    if not caches:
        raise BankError("no glyph caches found — run: python -m ocr_vlm.glyphs")
    bank = GlyphBank(size).fit(caches)
    styles = {str(c.get("style", "none")) for c in caches}
    bank.save(out or bank_path(data), meta={
        "sets": [c["set"] for c in caches],
        "style": ", ".join(sorted(styles)),
        "charset": CHARSET,
        "excluded_labels": dict(getattr(bank, "skipped_labels", {})),
    })
    return bank


def classify_raster(bank: GlyphBank, raster, *, blank: bool,
                    floor: float = GLYPH_BRIGHT_FLOOR
                    ) -> tuple[str, float, float, str]:
    """Canonical raster + blank flag -> (char, conf, margin, source).

    The blank-vs-glyph policy shared by every read path: no raster or a
    dim raster *that the ink test also called blank* is a blank cell
    (no classifier call); everything bright is classified, including
    cells the ink test called blank (the tiny '.'/'\'' glyphs).
    """
    if raster is None or not raster.any():
        return BLANK, 1.0, 1.0, "cv"
    if blank and float(np.percentile(raster, 99)) < floor:
        return BLANK, 1.0, 1.0, "cv"
    pred = bank.predict_raster(raster)
    if pred is None:
        return BLANK, 0.0, 0.0, "classifier"
    char, conf, margin = pred
    return char, conf, margin, "classifier"


def classify_cell(bank: GlyphBank, crop, *, blank: bool,
                  floor: float = GLYPH_BRIGHT_FLOOR
                  ) -> tuple[str, float, float, str]:
    """Live module crop -> gating + classification (read path)."""
    raster = segment.canonical_glyph(crop, glyphs.GLYPH_SIZE)
    return classify_raster(bank, raster, blank=blank, floor=floor)


# -- evaluation ----------------------------------------------------------------

def _photo_key(payload: dict, index: int) -> str:
    return f"{payload['set']}/{payload['photos'][index]}"


def _test_mask(keys: list[str], split: str, holdout: str | None,
               sets_of: list[str], photo_frac: float) -> np.ndarray:
    if split == "all":
        return np.ones(len(keys), dtype=bool)
    if split == "set":
        if not holdout:
            raise BankError("split='set' needs --holdout SET_NAME")
        mask = np.array([s == holdout for s in sets_of], dtype=bool)
        if not mask.any():
            raise BankError(f"no cells from set {holdout!r} in the caches")
        return mask
    if split == "photo":
        cutoff = int(round(max(0.0, min(0.9, photo_frac)) * 100))
        mask = np.array([zlib.crc32(k.encode()) % 100 < cutoff
                         for k in keys], dtype=bool)
        if not mask.any() or mask.all():
            raise BankError("photo split degenerated — not enough photos")
        return mask
    raise BankError(f"unknown split {split!r}")


def evaluate(data: str | None = None, *, sets: list[str] | None = None,
             size: int = DEFAULT_SIZE, split: str = "all",
             holdout: str | None = None, photo_frac: float = 0.2) -> dict:
    """Photo-disjoint / set-disjoint accuracy of the bank backend.

    Returns a metrics dict (also printed by the CLI): per-cell accuracy
    for the raw bank and for the reader simulation (blank gating + bank),
    row-exact rate and average mismatches per row, per-class recall and
    the top confusions.
    """
    caches = load_caches(data, sets)
    if not caches:
        raise BankError("no glyph caches found — run: python -m ocr_vlm.glyphs")
    crops = np.concatenate([c["crops"] for c in caches])
    labels = np.concatenate([c["labels"] for c in caches])
    blanks = np.concatenate([c["blanks"] for c in caches])
    keys: list[str] = []
    sets_of: list[str] = []
    positions: list[int] = []
    for payload in caches:
        for i in range(len(payload["labels"])):
            keys.append(_photo_key(payload, i))
            sets_of.append(payload["set"])
            positions.append(int(payload["positions"][i]))

    test = _test_mask(keys, split, holdout, sets_of, photo_frac)
    train_idx = np.flatnonzero(~test)
    test_idx = np.flatnonzero(test)
    if test.all():
        train_idx = test_idx  # in-sample sanity: train on the test cells
    if train_idx.size == 0:
        raise BankError("train split is empty")

    bank = GlyphBank(size).fit([
        {"crops": crops[train_idx], "labels": labels[train_idx]}])

    raw_pred = np.empty(len(test_idx), dtype=object)
    sim_pred = np.empty(len(test_idx), dtype=object)
    confs = np.empty(len(test_idx), dtype=np.float32)
    margins = np.empty(len(test_idx), dtype=np.float32)
    for out_i, cell in enumerate(test_idx):
        raw = bank.predict_raster(crops[cell])
        raw_char = raw[0] if raw else BLANK
        char, conf, margin, _source = classify_raster(
            bank, crops[cell], blank=bool(blanks[cell]))
        sim = char
        raw_pred[out_i] = raw_char
        sim_pred[out_i] = sim
        confs[out_i] = conf
        margins[out_i] = margin

    truth = labels[test_idx]
    raw_acc = float(np.mean(raw_pred == truth)) if len(test_idx) else 0.0
    sim_acc = float(np.mean(sim_pred == truth)) if len(test_idx) else 0.0

    # Row-level reconstruction: group the simulated reads by photo.
    rows: dict[str, list[tuple[int, str, str]]] = {}
    for out_i, cell in enumerate(test_idx):
        rows.setdefault(keys[cell], []).append(
            (positions[cell], str(sim_pred[out_i]), str(truth[out_i])))
    exact = 0
    mismatches = 0
    for cells in rows.values():
        cells.sort()
        got = "".join(c[1] for c in cells)
        want = "".join(c[2] for c in cells)
        mm = sum(1 for a, b in zip(got, want) if a != b)
        mismatches += mm
        if mm == 0:
            exact += 1

    # Confusions + per-class recall (reader simulation).
    per_class: dict[str, dict] = {}
    confusions: dict[tuple[str, str], int] = {}
    for out_i, cell in enumerate(test_idx):
        want, got = str(truth[out_i]), str(sim_pred[out_i])
        entry = per_class.setdefault(want, {"total": 0, "correct": 0})
        entry["total"] += 1
        if want == got:
            entry["correct"] += 1
        else:
            confusions[(want, got)] = confusions.get((want, got), 0) + 1
    top_confusions = sorted(confusions.items(), key=lambda kv: -kv[1])[:12]

    low = confs < 0.9
    return {
        "split": split,
        "holdout": holdout,
        "size": size,
        "train_cells": int(train_idx.size),
        "train_sets": sorted({sets_of[i] for i in train_idx}),
        "test_cells": int(test_idx.size),
        "test_photos": len(rows),
        "test_sets": sorted({sets_of[i] for i in test_idx}),
        "bank_classes": len(bank.classes),
        "bank_acc": round(raw_acc, 4),
        "sim_acc": round(sim_acc, 4),
        "rows": len(rows),
        "row_exact": exact,
        "row_exact_rate": round(exact / len(rows), 4) if rows else 0.0,
        "row_mismatches": mismatches,
        "row_mismatch_avg": round(mismatches / len(rows), 4) if rows else 0.0,
        "low_conf_cells": int(low.sum()),
        "low_conf_frac": round(float(low.mean()), 4) if len(low) else 0.0,
        "per_class": {k: {**v, "recall": round(v["correct"] / v["total"], 4)}
                      for k, v in sorted(per_class.items())},
        "confusions": [{"label": a, "pred": b, "count": n}
                       for (a, b), n in top_confusions],
    }


def print_eval(metrics: dict) -> None:
    print(f"split={metrics['split']}"
          + (f" holdout={metrics['holdout']}" if metrics.get("holdout") else "")
          + f" · size={metrics['size']} · bank classes={metrics['bank_classes']}")
    print(f"  train: {metrics['train_cells']} cells from "
          f"{', '.join(metrics['train_sets']) or '-'}")
    print(f"  test : {metrics['test_cells']} cells / {metrics['test_photos']} "
          f"photos from {', '.join(metrics['test_sets']) or '-'}")
    print(f"  per-cell: bank {metrics['bank_acc']:.3f} · "
          f"reader-sim {metrics['sim_acc']:.3f}")
    print(f"  rows: {metrics['row_exact']}/{metrics['rows']} exact "
          f"({metrics['row_exact_rate']:.3f}) · avg mismatches "
          f"{metrics['row_mismatch_avg']}")
    print(f"  low-confidence cells (<0.9): {metrics['low_conf_cells']} "
          f"({metrics['low_conf_frac']:.3f})")
    weak = [(c, v) for c, v in metrics["per_class"].items()
            if v["recall"] < 1.0]
    if weak:
        shown = ", ".join(f"{c!r} {v['recall']:.3f} ({v['correct']}/"
                          f"{v['total']})" for c, v in weak[:16])
        print(f"  classes below 1.0 recall: {shown}")
    if metrics["confusions"]:
        top = ", ".join(f"{c['label']!r}->{c['pred']!r} {c['count']}"
                        for c in metrics["confusions"][:8])
        print(f"  confusions: {top}")


# -- single-photo read (manual verification) -----------------------------------

def read_image(bank: GlyphBank, path: str, total: int = 12) -> dict:
    image = cv2.imread(path)
    if image is None:
        raise BankError(f"photo unreadable: {path}")
    display = segment.find_display(image)
    if display is None:
        raise BankError("display not detected")
    boxes = segment.module_boxes(image, display, total)
    cells = []
    for i, crop in enumerate(segment.crop_modules(image, boxes)):
        char, conf, margin, source = classify_cell(
            bank, crop, blank=segment.is_blank(crop))
        cells.append({"module": i, "char": char,
                      "confidence": round(conf, 3),
                      "margin": round(margin, 3), "source": source})
    return {"path": path, "display": display.as_dict(),
            "read": "".join(c["char"] for c in cells), "modules": cells}


# -- CLI -----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ocr_vlm.classifier",
        description="Local glyph classifier: build/evaluate the template bank")
    parser.add_argument("command", choices=["build", "eval", "read"])
    parser.add_argument("photo", nargs="?",
                        help="photo path for the read command")
    parser.add_argument("--data", default=None,
                        help="data folder (default: $OCR_VLM_DATA or ./data)")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE,
                        help=f"bank input size (default {DEFAULT_SIZE})")
    parser.add_argument("--sets", nargs="*", default=None,
                        help="only these glyph-cache sets (default: all)")
    parser.add_argument("--split", default="all",
                        choices=["all", "photo", "set"],
                        help="evaluation split (default all = in-sample)")
    parser.add_argument("--holdout", default=None,
                        help="baseline set to hold out for --split set")
    parser.add_argument("--photo-frac", type=float, default=0.2,
                        help="test fraction for --split photo (default 0.2)")
    parser.add_argument("--modules", type=int, default=12,
                        help="module count for the read command")
    args = parser.parse_args(argv)

    try:
        if args.command == "build":
            bank = train_bank(args.data, sets=args.sets, size=args.size)
            path = bank_path(args.data)
            print(f"trained bank: {path}")
            print(f"  classes: {len(bank.classes)} · cells: "
                  f"{sum(bank.counts.values())} · size: {bank.size}")
            print(f"  sets: {', '.join(bank.meta.get('sets', []))}")
            excluded = getattr(bank, "skipped_labels", {})
            if excluded:
                print(f"  excluded labels (not in charset): "
                      + ", ".join(f"{c!r} x{n}"
                                  for c, n in excluded.items()))
            for char, count in bank.counts.items():
                shown = "space" if char == " " else char
                print(f"    {shown!r}: {count}")
            return 0
        if args.command == "eval":
            metrics = evaluate(args.data, sets=args.sets, size=args.size,
                               split=args.split, holdout=args.holdout,
                               photo_frac=args.photo_frac)
            print_eval(metrics)
            return 0
        if not args.photo:
            parser.error("read needs a photo path")
        bank = GlyphBank.load(bank_path(args.data))
        result = read_image(bank, args.photo, total=args.modules)
        print(f"{result['path']}: {result['read']!r}")
        for cell in result["modules"]:
            if cell["source"] == "classifier":
                print(f"  [{cell['module']:2d}] {cell['char']!r} "
                      f"conf {cell['confidence']:.3f} margin "
                      f"{cell['margin']:.3f}")
        return 0
    except BankError as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
