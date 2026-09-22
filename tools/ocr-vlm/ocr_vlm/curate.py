"""Baseline set curation: ground-truth images + their actual content.

The benchmark's read-from-disk datasets pair each photo with ``want`` —
the frame the calibration app *commanded*. When the display itself is
the suspect (which is why the calibration tools exist), the photo may
show something other than ``want``, so ``want`` is not ground truth for
assessing an OCR model. A *baseline set* fixes that: photos are copied
into the set together with a human-verified ``content`` string, and the
benchmark scores models against that content instead.

A set lives under ``tools/ocr-vlm/baselines/<name>/`` (override with
``OCR_VLM_BASELINES``)::

    <name>/
      set.json        # name, created, source, module_count
      baseline.jsonl  # {"photo","content","status","prior_read","want","tag","frame_id"}
      images/         # copied photo files (originals, byte-identical)

``status`` is ``"pending"`` until a human saves the entry (saving marks
it ``"verified"``); the benchmark excludes pending entries by default so
pre-filled-but-unchecked rows are never mistaken for truth. ``content``
is pre-filled from whatever a previous run recorded (a calib-vlm run's
``read``, or a ``reads.jsonl`` ``saw``) purely as a typing aid.

Everything here is local file I/O — no network, no model.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone

from .dataset import valid_photo_name

SET_FILE = "set.json"
ENTRIES_FILE = "baseline.jsonl"
IMAGES_DIR = "images"
DEFAULT_MODULE_COUNT = 12

SET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

ENTRY_FIELDS = ("photo", "content", "status", "prior_read", "want", "tag",
                "frame_id")


class CurateError(RuntimeError):
    """A curation problem the caller should surface to the user (HTTP 4xx)."""


def baselines_root() -> str:
    """Folder holding every baseline set (env override for tests)."""
    override = os.environ.get("OCR_VLM_BASELINES")
    if override:
        return override
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "baselines")


def valid_set_name(name: str) -> bool:
    """Set names are one safe path segment (no separators, no dot-start)."""
    return bool(name) and bool(SET_NAME_RE.match(str(name)))


def set_dir(name: str) -> str:
    if not valid_set_name(name):
        raise CurateError(f"invalid set name {name!r}")
    return os.path.join(baselines_root(), str(name))


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
        raise CurateError(f"set {name!r} not found ({exc})") from exc
    except ValueError as exc:
        raise CurateError(f"set {name!r} has a corrupt {SET_FILE}: {exc}") \
            from exc
    if not isinstance(meta, dict):
        raise CurateError(f"set {name!r} has a corrupt {SET_FILE}")
    return meta


def read_entries(name: str) -> list[dict]:
    """All entries of a set, in file order (never raises on bad lines)."""
    path = _entries_path(name)
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError as exc:
        raise CurateError(f"set {name!r} has no {ENTRIES_FILE} ({exc})") \
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
    """Everything the curate UI needs for one set."""
    meta = _read_meta(name)
    entries = read_entries(name)
    return {"name": name, "meta": meta, "stats": stats(entries),
            "entries": entries}


def list_sets() -> list[dict]:
    """Summaries of every baseline set, newest creation first."""
    root = baselines_root()
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
        except CurateError:
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
        raise CurateError("status must be pending or verified")
    entries = read_entries(name)
    target = None
    for entry in entries:
        if entry.get("photo") == photo:
            target = entry
            break
    if target is None:
        raise CurateError(f"{photo!r} is not in set {name!r}")
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
        raise CurateError(f"{photo!r} is not in set {name!r}")
    path = os.path.join(_images_path(name), os.path.basename(str(photo)))
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass
    _write_entries(name, kept)
    return stats(kept)


def delete_set(name: str) -> None:
    """Discard a whole baseline set (folder, images, entries)."""
    path = set_dir(name)
    if not os.path.isdir(path):
        raise CurateError(f"set {name!r} not found")
    shutil.rmtree(path)


# -- creating from a source run ----------------------------------------------

def _source_items(source_dir: str) -> list[dict]:
    """Normalized ``{photo, want, prior_read, tag, frame_id}`` items.

    Accepts a calib-vlm run dir (``report.json`` frames — ``want`` is the
    commanded frame, ``prior_read`` the recorded read) or a dataset dir
    with ``reads.jsonl`` (``saw`` drives the pre-fill). ``reads.jsonl``
    wins when both exist.
    """
    reads = os.path.join(source_dir, "reads.jsonl")
    report = os.path.join(source_dir, "report.json")
    if os.path.isfile(reads):
        items: list[dict] = []
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
                              "frame_id": item.get("frameId")})
        return items
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
                          "frame_id": frame.get("frameId")})
        if items:
            return items
    raise CurateError(
        f"no reads.jsonl or report.json frames found in {source_dir}")


def create_set(source_dir: str, name: str | None = None,
               module_count: int = DEFAULT_MODULE_COUNT) -> dict:
    """Import a source run into a new baseline set (images copied, pending).

    ``content`` is pre-filled with the previous run's read purely so the
    curator has a starting string to correct; every entry starts
    ``pending`` and is excluded from benchmarks until verified.
    """
    source_dir = os.path.abspath(str(source_dir or ""))
    if not os.path.isdir(source_dir):
        raise CurateError(f"source directory not found: {source_dir}")
    items = _source_items(source_dir)
    if not items:
        raise CurateError(f"no frames found in {source_dir}")

    if not name:
        name = os.path.basename(os.path.normpath(source_dir))
    name = str(name).strip()
    if not valid_set_name(name):
        raise CurateError(
            "set name must be 1-64 chars of letters, digits, '.', '_' or '-'")
    target = set_dir(name)
    if os.path.exists(target):
        raise CurateError(f"set {name!r} already exists — pick another name")

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
            source_photo = os.path.join(source_dir, photo)
            if not os.path.isfile(source_photo):
                skipped.append(photo)
                continue
            shutil.copy2(source_photo, os.path.join(images_dir, photo))
            entries.append({
                "photo": photo,
                "content": item["prior_read"],
                "status": "pending",
                "prior_read": item["prior_read"],
                "want": item["want"],
                "tag": item["tag"],
                "frame_id": item["frame_id"],
            })
        if not entries:
            raise CurateError("every source photo is missing on disk")
        _write_entries(name, entries)
        _write_json_atomic(os.path.join(target, SET_FILE), {
            "name": name,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source_dir,
            "module_count": int(module_count),
        })
    except Exception:
        shutil.rmtree(target, ignore_errors=True)  # never leave a half set
        raise
    return {"name": name, "path": target, "count": len(entries),
            "skipped": skipped, "stats": stats(entries)}
