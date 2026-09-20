"""Glyph crop dataset built from curated baseline sets.

The VLM benchmark established that off-the-shelf models — OCR
specialists and general vision models alike — misread this display's
font in consistent, font-specific ways (``0`` read as ``O``, ``/`` as
``I``, ``1`` as ``7``). The font is *not* ambiguous (the zero glyph is
dotted, the one is flagged); the models are simply out of distribution.
The fix is a recognizer trained on the font itself, and the curated
baseline sets are its training data: every verified entry pairs a photo
with the human-checked content, so each module crop is a labeled glyph
sample.

This module extracts those samples (Stage A of the classifier plan):

- walk ``curate.list_sets()`` (only ``verified`` entries; pending
  pre-fills are typing aids, never labels),
- locate the display per photo (``segment.find_display``), split it on
  the set's module count (``segment.module_boxes``) and crop each
  module (``segment.crop_modules``),
- resize each crop to a fixed grayscale square (``GLYPH_SIZE``),
  optionally styled (``segment.apply_style``), and label it with the
  curated character at that position (``content[i]``),
- record the blank decision (``segment.is_blank``) so evaluation can
  simulate the reader's blank gating without re-thresholding resized
  crops.

Photos that cannot contribute are skipped *with a reason* (no display
detected, missing/unreadable file, content width != module count) and
the counts are kept in the cache summary, so the dataset's coverage is
auditable.

Output lives under ``<data>/glyphs/`` (``OCR_VLM_DATA``, default
``data/`` in the working directory) as one ``<set>.npz`` plus a
``<set>.json`` summary per set. Pure file I/O + OpenCV, no model, no
network.

CLI::

    python -m ocr_vlm.glyphs [--data DIR] [--style none|contrast|invert|binary]
                             [--sets name ...]
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import cv2
import numpy as np

from . import curate, segment

CACHE_DIRNAME = "glyphs"
# Stored crop size in px (square). Crops are resized here once so every
# consumer (bank, CNN) trains on the same canonical pixels; classifiers
# may resize further (INTER_AREA) without going back to the photos.
GLYPH_SIZE = 64
DATASET_VERSION = 1
# npz files in the cache folder that are NOT per-set crop caches (the
# trained bank lives alongside the caches; listing it as a set would
# try to read its template arrays as crops).
RESERVED_NPZ = {"bank"}


def data_root() -> str:
    """Folder holding tool data (same convention as the server)."""
    override = os.environ.get("OCR_VLM_DATA")
    return override or os.path.join(os.getcwd(), "data")


def cache_dir(data: str | None = None) -> str:
    return os.path.join(data or data_root(), CACHE_DIRNAME)


def cache_paths(data: str | None, set_name: str) -> tuple[str, str]:
    base = os.path.join(cache_dir(data), set_name)
    return base + ".npz", base + ".json"


def _class_counts(labels) -> dict:
    counts: dict[str, int] = {}
    for label in labels:
        key = "space" if label == " " else str(label)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _glyph_gray(crop, style: str, size: int) -> np.ndarray | None:
    """Module crop -> canonical glyph raster (see segment.canonical_glyph)."""
    return segment.canonical_glyph(crop, size, style=style)


def extract_set(set_name: str, *, module_count: int | None = None,
                style: str = "none",
                size: int = GLYPH_SIZE) -> dict:
    """Labeled crops from every verified entry of one baseline set.

    Returns a payload dict: ``crops`` uint8 (N, size, size), ``labels``
    str, ``blanks`` bool, ``photos`` str, ``positions`` int, plus the
    set metadata and the skip report. Never raises for a single bad
    photo; a photo that cannot contribute is recorded in ``skipped``.
    """
    if style not in segment.PREPROCESS_STYLES:
        raise ValueError(f"unknown style {style!r} "
                         f"(choose from {', '.join(segment.PREPROCESS_STYLES)})")
    info = curate.get_set(set_name)
    meta = info.get("meta") or {}
    total = int(module_count or meta.get("module_count")
                or curate.DEFAULT_MODULE_COUNT)
    crops: list[np.ndarray] = []
    labels: list[str] = []
    blanks: list[bool] = []
    photos: list[str] = []
    positions: list[int] = []
    skipped: dict[str, str] = {}
    checked = 0
    for entry in info["entries"]:
        if entry.get("status") != "verified":
            continue
        checked += 1
        photo = str(entry.get("photo") or "")
        content = str(entry.get("content") or "")
        if len(content) != total:
            skipped[photo] = (f"content width {len(content)} != "
                              f"module count {total}")
            continue
        path = curate.photo_file(set_name, photo)
        if path is None:
            skipped[photo] = "photo file missing"
            continue
        image = cv2.imread(path)
        if image is None:
            skipped[photo] = "photo unreadable"
            continue
        display = segment.find_display(image)
        if display is None:
            skipped[photo] = "display not detected"
            continue
        boxes = segment.module_boxes(image, display, total)
        crops_at = segment.crop_modules(image, boxes)
        if len(crops_at) != total:
            skipped[photo] = f"crop count {len(crops_at)} != {total}"
            continue
        row_crops: list[np.ndarray] = []
        for crop in crops_at:
            gray = _glyph_gray(crop, style, size)
            if gray is None:
                # No ink at all: a genuinely blank cell (or a failed
                # segment). Keep the photo — a blank-cell raster is a
                # valid sample (' ' class); dropping the whole photo
                # over one blank cell would throw away 11 good labels.
                gray = np.zeros((int(size), int(size)), np.uint8)
            row_crops.append(gray)
        for i, crop in enumerate(crops_at):
            crops.append(row_crops[i])
            labels.append(content[i])
            blanks.append(bool(segment.is_blank(crop)))
            photos.append(photo)
            positions.append(i)
    stack = (np.stack(crops) if crops
             else np.zeros((0, size, size), np.uint8))
    payload = {
        "set": set_name,
        "module_count": total,
        "style": style,
        "size": int(size),
        "crops": stack,
        "labels": np.array(labels, dtype=object).astype(str),
        "blanks": np.array(blanks, dtype=bool),
        "photos": np.array(photos, dtype=object).astype(str),
        "positions": np.array(positions, dtype=np.int32),
        "summary": {
            "set": set_name,
            "module_count": total,
            "style": style,
            "size": int(size),
            "verified_entries": checked,
            "cells": len(labels),
            "photos": len(set(photos)),
            "skipped": len(skipped),
            "skipped_reasons": skipped,
            "blank_cells": int(sum(1 for b in blanks if b)),
            "counts": _class_counts(labels),
        },
    }
    payload["meta"] = {"created": datetime.now(timezone.utc).isoformat(
        timespec="seconds"), "version": DATASET_VERSION}
    return payload


def save_cache(data: str | None, payload: dict) -> tuple[str, str]:
    """Write one set's cache (npz + json summary); returns both paths."""
    directory = cache_dir(data)
    os.makedirs(directory, exist_ok=True)
    npz_path, json_path = cache_paths(data, payload["set"])
    tmp = npz_path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            crops=payload["crops"], labels=payload["labels"],
            blanks=payload["blanks"], photos=payload["photos"],
            positions=payload["positions"],
            module_count=int(payload["module_count"]),
            style=payload["style"], size=int(payload["size"]))
    os.replace(tmp, npz_path)
    summary = dict(payload["summary"])
    summary.update({"created": payload["meta"]["created"],
                    "version": payload["meta"]["version"],
                    "file": os.path.basename(npz_path)})
    tmp = json_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    os.replace(tmp, json_path)
    return npz_path, json_path


def load_cache(data: str | None, set_name: str) -> dict:
    """Read a set's cache back (payload shape of ``extract_set``)."""
    npz_path, json_path = cache_paths(data, set_name)
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"no glyph cache at {npz_path}")
    with np.load(npz_path, allow_pickle=False) as npz:
        payload = {
            "set": set_name,
            "crops": npz["crops"],
            "labels": npz["labels"].astype(str),
            "blanks": npz["blanks"],
            "photos": npz["photos"].astype(str),
            "positions": npz["positions"],
            "module_count": int(npz["module_count"]),
            "style": str(npz["style"]),
            "size": int(npz["size"]),
        }
    summary = {}
    if os.path.isfile(json_path):
        try:
            with open(json_path, encoding="utf-8") as fh:
                summary = json.load(fh)
        except (OSError, ValueError):
            summary = {}
    payload["summary"] = summary
    payload["meta"] = {"created": summary.get("created", ""),
                       "version": summary.get("version", DATASET_VERSION)}
    return payload


def available_sets(data: str | None = None) -> list[str]:
    directory = cache_dir(data)
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return []
    return [os.path.splitext(n)[0] for n in names
            if n.endswith(".npz")
            and os.path.splitext(n)[0] not in RESERVED_NPZ]


def build(data: str | None = None, *, style: str = "none",
          sets: list[str] | None = None) -> dict:
    """Extract + save caches for every (requested) baseline set.

    Sets with zero verified entries are skipped (nothing to learn from
    uncurated pre-fills) and reported. Returns an aggregate summary for
    the CLI and tests.
    """
    result: dict = {"style": style, "sets": {}, "cells": 0, "photos": 0,
                    "skipped": 0, "no_train_set": []}
    requested = set(sets) if sets else None
    for info in curate.list_sets():
        name = info["name"]
        if requested is not None and name not in requested:
            continue
        if info["stats"]["verified"] == 0:
            result["no_train_set"].append(name)
            continue
        payload = extract_set(name, style=style)
        if payload["summary"]["cells"] == 0:
            result["no_train_set"].append(name)
            result["sets"][name] = payload["summary"]
            continue
        save_cache(data, payload)
        result["sets"][name] = payload["summary"]
        result["cells"] += payload["summary"]["cells"]
        result["photos"] += payload["summary"]["photos"]
        result["skipped"] += payload["summary"]["skipped"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ocr_vlm.glyphs",
        description="Build labeled glyph-crop caches from curated baseline sets")
    parser.add_argument("--data", default=None,
                        help="data folder (default: $OCR_VLM_DATA or ./data)")
    parser.add_argument("--style", default="none",
                        choices=list(segment.PREPROCESS_STYLES),
                        help="preprocessing style applied before caching")
    parser.add_argument("--sets", nargs="*", default=None,
                        help="only these baseline set names (default: all)")
    args = parser.parse_args(argv)

    result = build(args.data, style=args.style, sets=args.sets)
    if not result["sets"]:
        print("no baseline sets found (or none requested)")
        return 1
    print(f"glyph caches in {cache_dir(args.data)}")
    for name, summary in result["sets"].items():
        line = (f"  {name}: {summary['cells']} cells from "
                f"{summary['photos']} photos")
        if summary.get("skipped"):
            line += f", {summary['skipped']} skipped"
        if not summary.get("cells"):
            line += " (no verified entries with detectable display)"
        print(line)
        for photo, reason in list(summary.get("skipped_reasons", {}).items())[:5]:
            print(f"      - {photo}: {reason}")
    print(f"total: {result['cells']} cells, {result['photos']} photos, "
          f"{result['skipped']} skipped")
    if result["no_train_set"]:
        print(f"no usable entries: {', '.join(result['no_train_set'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
