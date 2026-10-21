"""Golden set curation: labeled photos + their verified content.

The golden set is the ground truth of the whole tool: the benchmark
scores models against its content, and the CNN trains on its photos.
Every entry pairs a photo with a human-verified content string; entries
start ``pending`` (pre-filled from whatever a previous reader reported)
and become ``verified`` when a human saves them. Benchmarks and
training only ever use verified entries, so a half-curated set can
never silently pass pre-fills off as truth.

A set lives under ``<base>/golden/<name>/`` (git-tracked)::

    <name>/
      set.json       # name, created, source_format, module_count
      labels.jsonl   # {"photo","content","status","prior_read","want","tag","frame_id"}
      images/        # photo files (recompressed on import)

Imports accept the formats the older tools produced — a golden/legacy
``labels.jsonl``/``baseline.jsonl`` set (content + status preserved),
a ``reads.jsonl`` dataset (``saw`` drives the pre-fill) or a calib-vlm
run's ``report.json`` frames (``read`` drives it). Photos are
recompressed on import (lossless PNG level 9 by default; JPEG q92 when
a set's PNG total exceeds the budget) so the tracked golden set stays
a reasonable git size while remaining the exact pixels training and
benchmarking see.

Everything here is local file I/O — no network, no model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone

import cv2

from . import paths
from .dataset import GOLDEN_FILE, LEGACY_FILES, photo_path, valid_photo_name

SET_FILE = "set.json"
ENTRIES_FILE = GOLDEN_FILE
IMAGES_DIR = "images"
DEFAULT_MODULE_COUNT = 12
DEFAULT_PNG_BUDGET_MB = 15.0
JPEG_QUALITY = 92

SET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

ENTRY_FIELDS = ("photo", "content", "status", "prior_read", "want", "tag",
                "frame_id")


class GoldenError(RuntimeError):
    """A curation problem the caller should surface to the user (HTTP 4xx)."""


def valid_set_name(name: str) -> bool:
    """Set names are one safe path segment (no separators, no dot-start)."""
    return bool(name) and bool(SET_NAME_RE.match(str(name)))


def set_dir(name: str) -> str:
    if not valid_set_name(name):
        raise GoldenError(f"invalid set name {name!r}")
    return os.path.join(paths.golden_root(), str(name))


def _entries_path(name: str) -> str:
    return os.path.join(set_dir(name), ENTRIES_FILE)


def _images_path(name: str) -> str:
    return os.path.join(set_dir(name), IMAGES_DIR)


def photo_file(name: str, photo: str) -> str | None:
    """Resolve a photo inside a set's ``images/`` folder (None if absent)."""
    if not valid_photo_name(photo):
        return None
    path = os.path.join(_images_path(name), os.path.basename(str(photo)))
    return path if os.path.isfile(path) else None


# -- reading ------------------------------------------------------------------

def _read_meta(name: str) -> dict:
    path = os.path.join(set_dir(name), SET_FILE)
    try:
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except OSError as exc:
        raise GoldenError(f"set {name!r} not found ({exc})") from exc
    except ValueError as exc:
        raise GoldenError(f"set {name!r} has a corrupt {SET_FILE}: {exc}") \
            from exc
    if not isinstance(meta, dict):
        raise GoldenError(f"set {name!r} has a corrupt {SET_FILE}")
    return meta


def read_entries(name: str) -> list[dict]:
    """All entries of a set, in file order (never raises on bad lines)."""
    path = _entries_path(name)
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError as exc:
        raise GoldenError(f"set {name!r} has no {ENTRIES_FILE} ({exc})") \
            from exc
    entries: list[dict] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and item.get("photo"):
            entries.append(item)
    return entries


def stats(entries: list[dict]) -> dict:
    verified = sum(1 for e in entries if e.get("status") == "verified")
    return {"total": len(entries), "verified": verified,
            "pending": len(entries) - verified}


def get_set(name: str) -> dict:
    """Everything the golden UI needs for one set."""
    meta = _read_meta(name)
    entries = read_entries(name)
    return {"name": name, "meta": meta, "stats": stats(entries),
            "entries": entries}


def list_sets() -> list[dict]:
    """Summaries of every golden set, newest creation first."""
    root = paths.golden_root()
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return []
    out: list[dict] = []
    for name in names:
        if not valid_set_name(name):
            continue
        if not os.path.isfile(os.path.join(root, name, ENTRIES_FILE)):
            continue
        try:
            info = get_set(name)
        except GoldenError:
            continue
        out.append({"name": name, "meta": info["meta"],
                    "stats": info["stats"],
                    "path": set_dir(name)})
    out.sort(key=lambda s: str(s["meta"].get("created", "")), reverse=True)
    return out


# -- writing ------------------------------------------------------------------

def _write_json_atomic(path: str, payload) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def _write_entries(name: str, entries: list[dict]) -> None:
    path = _entries_path(name)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for entry in entries:
            record = {key: entry.get(key) for key in ENTRY_FIELDS}
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def update_entry(name: str, photo: str, content: str | None = None,
                 status: str | None = None) -> dict:
    """Save one entry's content and/or status (saving content = verified)."""
    if status is not None and status not in ("pending", "verified"):
        raise GoldenError("status must be pending or verified")
    entries = read_entries(name)
    target = None
    for entry in entries:
        if entry.get("photo") == photo:
            target = entry
            break
    if target is None:
        raise GoldenError(f"{photo!r} is not in set {name!r}")
    if content is not None:
        # A one-line string only: fold any newline into a space (the
        # display content is a single row; newlines are edit accidents).
        target["content"] = str(content).replace("\r", " ").replace("\n", " ")
    if status is not None:
        target["status"] = status
    elif content is not None:
        target["status"] = "verified"   # typing the truth IS the verification
    _write_entries(name, entries)
    return target


def remove_image(name: str, photo: str) -> dict:
    """Drop one image (entry + copied file) from a set."""
    entries = read_entries(name)
    kept = [e for e in entries if e.get("photo") != photo]
    if len(kept) == len(entries):
        raise GoldenError(f"{photo!r} is not in set {name!r}")
    path = os.path.join(_images_path(name), os.path.basename(str(photo)))
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
    _write_entries(name, kept)
    return stats(kept)


def delete_set(name: str) -> None:
    """Discard a whole golden set (folder, images, entries)."""
    path = set_dir(name)
    if not os.path.isdir(path):
        raise GoldenError(f"set {name!r} not found")
    shutil.rmtree(path)


# -- importing from a source run ------------------------------------------------

def _source_items(source_dir: str) -> tuple[list[dict], str]:
    """Normalized import items + a human name for the source format.

    Item shape: ``{photo, want, prior_read, tag, frame_id, content?,
    status?}``. ``content``/``status`` are present when the source
    already carries curated truth (a golden/labels set) — those are
    preserved so importing never downgrades verified data to pending
    pre-fills. Formats, in priority order:

    - ``labels.jsonl`` / ``baseline.jsonl``: a curated set (content +
      status preserved; ``prior_read`` kept as the baseline column)
    - ``reads.jsonl``: a benchmark dataset (``saw`` drives the pre-fill)
    - ``report.json``: a calib-vlm run (``read`` drives the pre-fill)
    """
    for name in (GOLDEN_FILE, *LEGACY_FILES):
        path = os.path.join(source_dir, name)
        if os.path.isfile(path):
            items = []
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(item, dict) or not item.get("photo"):
                        continue
                    items.append({
                        "photo": str(item["photo"]),
                        "want": str(item.get("want") or ""),
                        "prior_read": str(item.get("prior_read") or ""),
                        "tag": str(item.get("tag") or ""),
                        "frame_id": item.get("frame_id"),
                        "content": (str(item["content"])
                                    if item.get("content") is not None
                                    else None),
                        "status": str(item.get("status") or "pending"),
                    })
            if items:
                return items, name
    reads = os.path.join(source_dir, "reads.jsonl")
    if os.path.isfile(reads):
        items = []
        with open(reads, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(item, dict) or not item.get("photo"):
                    continue
                saw = item.get("saw") or ""
                items.append({"photo": str(item["photo"]),
                              "want": str(item.get("want") or ""),
                              "prior_read": str(saw),
                              "tag": str(item.get("tag") or ""),
                              "frame_id": item.get("frameId"),
                              "content": None, "status": "pending"})
        if items:
            return items, "reads.jsonl"
    report = os.path.join(source_dir, "report.json")
    if os.path.isfile(report):
        with open(report, encoding="utf-8") as fh:
            payload = json.load(fh)
        frames = payload.get("frames") if isinstance(payload, dict) else None
        items = []
        for frame in frames or []:
            if not isinstance(frame, dict) or not frame.get("photo"):
                continue
            read = frame.get("read") or ""
            items.append({"photo": str(frame["photo"]),
                          "want": str(frame.get("frame") or ""),
                          "prior_read": str(read),
                          "tag": str(frame.get("tag") or ""),
                          "frame_id": frame.get("frameId"),
                          "content": None, "status": "pending"})
        if items:
            return items, "report.json"
    raise GoldenError(
        f"no {GOLDEN_FILE}, reads.jsonl or report.json frames found in "
        f"{source_dir}")


def _encode_sizes(source_dir: str, photos: list[str]) -> tuple[int, int]:
    """(png_total, jpeg_total) bytes for the photos, encoded in memory."""
    png_total = jpeg_total = 0
    for photo in photos:
        path = photo_path(source_dir, photo)
        if path is None:
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        ok, buf = cv2.imencode(".png", img,
                               [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
        if ok:
            png_total += len(buf)
        ok, buf = cv2.imencode(".jpg", img,
                               [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            jpeg_total += len(buf)
    return png_total, jpeg_total


def _recompressed_name(photo: str, fmt: str) -> str:
    stem = os.path.splitext(os.path.basename(photo))[0]
    return stem + (".jpg" if fmt == "jpeg" else ".png")


def _write_image(img, path: str, fmt: str) -> bool:
    if fmt == "jpeg":
        return bool(cv2.imwrite(path, img,
                                [int(cv2.IMWRITE_JPEG_QUALITY),
                                 JPEG_QUALITY]))
    return bool(cv2.imwrite(path, img,
                            [int(cv2.IMWRITE_PNG_COMPRESSION), 9]))


def create_set(source_dir: str, name: str | None = None,
               module_count: int = DEFAULT_MODULE_COUNT,
               image_format: str = "auto",
               png_budget_mb: float = DEFAULT_PNG_BUDGET_MB) -> dict:
    """Import a source run into a new golden set.

    Entries carry over their verified status and content when the
    source already has curated truth; otherwise ``content`` is
    pre-filled with the previous reader's answer purely so the curator
    has a starting string to correct, and every entry starts
    ``pending`` (excluded from benchmarks and training until verified).

    ``image_format``: ``auto`` (PNG level 9, falling back to JPEG q92
    when the set's PNG total exceeds ``png_budget_mb``), ``png``,
    ``jpeg``, or ``keep`` (byte-identical copies — used by tests).
    Photos are decoded and re-encoded, so the tracked golden set stays
    a sane git size; training and benchmarking always read the stored
    copies, keeping the two consistent.
    """
    if image_format not in ("auto", "png", "jpeg", "keep"):
        raise GoldenError("image_format must be auto, png, jpeg or keep")
    source_dir = os.path.abspath(str(source_dir or ""))
    if not os.path.isdir(source_dir):
        raise GoldenError(f"source directory not found: {source_dir}")
    items, source_format = _source_items(source_dir)
    if not items:
        raise GoldenError(f"no frames found in {source_dir}")

    if not name:
        name = os.path.basename(os.path.normpath(source_dir))
    name = str(name).strip()
    if not valid_set_name(name):
        raise GoldenError(
            "set name must be 1-64 chars of letters, digits, '.', '_' or '-'")
    target = set_dir(name)
    if os.path.exists(target):
        raise GoldenError(f"set {name!r} already exists — pick another name")

    # Decide the storage format up front (one extra decode pass, once).
    fmt = image_format
    encoding = {"png_bytes": None, "jpeg_bytes": None}
    if fmt == "auto":
        photos = [item["photo"] for item in items
                  if photo_path(source_dir, item["photo"])]
        png_total, jpeg_total = _encode_sizes(source_dir, photos)
        encoding["png_bytes"] = png_total
        encoding["jpeg_bytes"] = jpeg_total
        fmt = "png" if png_total <= int(png_budget_mb * 1024 * 1024) \
            else "jpeg"

    images_dir = _images_path(name)
    os.makedirs(images_dir, exist_ok=False)
    entries: list[dict] = []
    skipped: list[str] = []
    seen: set[str] = set()
    try:
        for item in items:
            photo = item["photo"]
            if photo in seen or not valid_photo_name(photo):
                continue
            seen.add(photo)
            source_photo = photo_path(source_dir, photo)
            if source_photo is None:
                skipped.append(photo)
                continue
            if fmt == "keep":
                new_photo = photo
                shutil.copy2(source_photo,
                             os.path.join(images_dir, new_photo))
            else:
                img = cv2.imread(source_photo)
                if img is None:
                    skipped.append(photo)
                    continue
                new_photo = _recompressed_name(photo, fmt)
                if not _write_image(img, os.path.join(images_dir, new_photo),
                                    fmt):
                    skipped.append(photo)
                    continue
            content = item.get("content")
            status = item.get("status") or "pending"
            if content is None:
                content = item["prior_read"]      # pre-fill typing aid
                status = "pending"
            entries.append({
                "photo": new_photo,
                "content": content,
                "status": status,
                "prior_read": item["prior_read"],
                "want": item["want"],
                "tag": item["tag"],
                "frame_id": item["frame_id"],
            })
        if not entries:
            raise GoldenError("every source photo is missing on disk")
        _write_entries(name, entries)
        _write_json_atomic(os.path.join(target, SET_FILE), {
            "name": name,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_format": source_format,
            "module_count": int(module_count),
            "image_format": fmt,
        })
    except Exception:
        shutil.rmtree(target, ignore_errors=True)  # never leave a half set
        raise
    return {"name": name, "path": target, "count": len(entries),
            "skipped": skipped, "stats": stats(entries),
            "image_format": fmt, "encoding": encoding}


# -- CLI -----------------------------------------------------------------------

def _cmd_list(_args) -> int:
    sets = list_sets()
    if not sets:
        print(f"no golden sets under {paths.golden_root()}")
        return 0
    for info in sets:
        st = info["stats"]
        meta = info["meta"]
        print(f"  {info['name']}: {st['verified']}/{st['total']} verified "
              f"({st['pending']} pending) · imported {meta.get('created', '?')}"
              f" · format {meta.get('image_format', '?')}")
    return 0


def _cmd_import(args) -> int:
    result = create_set(args.source, name=args.name,
                        module_count=args.modules,
                        image_format=args.format,
                        png_budget_mb=args.budget_mb)
    enc = result["encoding"]
    if enc["png_bytes"] is not None:
        print(f"encoding probe: png {enc['png_bytes'] / 1e6:.1f} MB vs "
              f"jpeg {enc['jpeg_bytes'] / 1e6:.1f} MB → chose "
              f"{result['image_format']}")
    print(f"imported {result['count']} entries into "
          f"{result['path']} ({result['image_format']})")
    st = result["stats"]
    print(f"  verified {st['verified']} · pending {st['pending']}")
    if result["skipped"]:
        print(f"  skipped {len(result['skipped'])}: "
              + ", ".join(result["skipped"][:5]))
    total = sum(os.path.getsize(os.path.join(result["path"], IMAGES_DIR, f))
                for f in os.listdir(os.path.join(result["path"], IMAGES_DIR)))
    print(f"  images total {total / 1e6:.1f} MB")
    return 0


def _cmd_stats(args) -> int:
    info = get_set(args.name)
    st = info["stats"]
    print(f"{args.name}: {st['verified']}/{st['total']} verified")
    counts: dict[str, int] = {}
    for entry in info["entries"]:
        if entry.get("status") != "verified":
            continue
        for ch in str(entry.get("content") or ""):
            counts[ch] = counts.get(ch, 0) + 1
    for ch in sorted(counts):
        shown = "space" if ch == " " else ch
        print(f"  {shown!r}: {counts[ch]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calib_auto.golden",
        description="Golden set curation: import, list, inspect")
    sub = parser.add_subparsers(dest="command", required=True)
    p_list = sub.add_parser("list", help="list golden sets")
    p_list.set_defaults(func=_cmd_list)
    p_import = sub.add_parser("import", help="import a source run as a set")
    p_import.add_argument("--source", required=True)
    p_import.add_argument("--name", default=None)
    p_import.add_argument("--modules", type=int, default=DEFAULT_MODULE_COUNT)
    p_import.add_argument("--format", default="auto",
                          choices=["auto", "png", "jpeg", "keep"])
    p_import.add_argument("--budget-mb", type=float,
                          default=DEFAULT_PNG_BUDGET_MB)
    p_import.set_defaults(func=_cmd_import)
    p_stats = sub.add_parser("stats", help="per-character counts of a set")
    p_stats.add_argument("name")
    p_stats.set_defaults(func=_cmd_stats)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except GoldenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
