"""Dataset loading and validation for the ocr-vlm benchmark.

A dataset is a JSONL file (default ``reads.jsonl``) whose lines look
like::

    {"photo": "sw_37_f255.png", "want": "%%%%%%%%%%%%", "saw": "%%%%%%%#%%%%"}

``photo`` names an image in the SAME directory as the JSONL file, ``want``
is the frame that was commanded when the photo was taken and ``saw`` is
what the reader under test (the baseline) reported back then. Everything
in this module is read-only: it parses, validates and resolves paths and
never writes.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field

PHOTO_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# Issue wording is a contract with describe()/tests: keep it a constant.
ISSUE_PHOTO = "photo not found"
ISSUE_DUPLICATE = "duplicate photo name"

# Sibling tool data roots (relative to tools/) scanned by "Discover":
# the known reads.jsonl samples live in previous calib-vlm run dirs.
_DISCOVER_ROOTS = (
    ("calib-vlm", "data", "runs"),
    ("calib-agent", "app", "data", "runs"),
)


@dataclass
class Record:
    """One dataset line, kept even when it cannot be run."""

    index: int
    photo: str
    want: str
    saw: str = ""
    fatal: list[str] = field(default_factory=list)   # row cannot be run
    issues: list[str] = field(default_factory=list)  # row runs, but flagged

    @property
    def runnable(self) -> bool:
        return not self.fatal

    @property
    def width(self) -> int:
        return len(self.want)


@dataclass
class Dataset:
    directory: str
    filename: str
    records: list[Record] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)  # malformed lines etc.
    missing_photos: list[str] = field(default_factory=list)

    @property
    def path(self) -> str:
        return os.path.join(self.directory, self.filename)

    @property
    def runnable(self) -> list[Record]:
        return [r for r in self.records if r.runnable]


def valid_photo_name(name: str) -> bool:
    """Bare file name with an image extension (no separators, no games).

    Windows tolerates both "/" and "\\" as separators, so a naive
    basename check is not enough: only a name that is exactly its own
    basename AND has no separator characters may be served.
    """
    if (not name or "\\" in name or "/" in name
            or name != os.path.basename(name) or name.startswith(".")):
        return False
    return name.lower().endswith(PHOTO_EXTS)


def photo_path(directory: str, name: str) -> str | None:
    """Resolve a photo inside the dataset directory (None when missing)."""
    if not valid_photo_name(name):
        return None
    path = os.path.join(directory, name)
    return path if os.path.isfile(path) else None


def prefix_of(photo: str) -> str:
    """Tag family of a photo name (``sw_37_f255.png`` -> ``sw``).

    The dataset groups run in tagged families (``sw_``, ``p0_``,
    ``ladder_``, ...); the raw prefix is the part before the first
    underscore, which is exactly the family for every known tag.
    """
    stem = os.path.splitext(os.path.basename(photo))[0]
    return stem.split("_")[0] or "other"


def load_dataset(directory: str, filename: str = "reads.jsonl",
                 module_count: int = 12) -> Dataset:
    """Parse and validate a dataset file (photos are checked on disk).

    Malformed lines become dataset-level ``problems``. Missing fields,
    bad widths and missing photo files become per-record ``fatal``/
    ``issues`` markers so every line shows up in the UI with a reason
    instead of being silently dropped.
    """
    directory = str(directory or "")
    filename = str(filename or "reads.jsonl")
    ds = Dataset(directory=directory, filename=filename)
    path = ds.path
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError as exc:
        ds.problems.append(f"cannot read {path}: {exc}")
        return ds
    seen: dict[str, int] = {}
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError as exc:
            ds.problems.append(f"line {lineno}: invalid JSON: {exc}")
            continue
        if not isinstance(item, dict):
            ds.problems.append(f"line {lineno}: not a JSON object")
            continue
        rec = _record_from_item(len(ds.records), lineno, item, module_count)
        if rec.photo in seen:
            rec.issues.append(
                f"{ISSUE_DUPLICATE} (first at line {seen[rec.photo]})")
        else:
            seen[rec.photo] = lineno
        if rec.runnable and rec.photo:
            if not os.path.isfile(os.path.join(directory, rec.photo)):
                rec.issues.append(ISSUE_PHOTO)
                if rec.photo not in ds.missing_photos:
                    ds.missing_photos.append(rec.photo)
        ds.records.append(rec)
    return ds


def _record_from_item(index: int, lineno: int, item: dict,
                      module_count: int) -> Record:
    photo = item.get("photo")
    want = item.get("want")
    saw = item.get("saw", "")
    fatal: list[str] = []
    issues: list[str] = []
    if not isinstance(photo, str) or not photo.strip():
        fatal.append(f"line {lineno}: missing photo name")
        photo = ""
    elif not valid_photo_name(photo):
        fatal.append(f"line {lineno}: bad photo name {photo!r}")
    if not isinstance(want, str) or not want:
        fatal.append(f"line {lineno}: missing want string")
        want = ""
    if saw is None:
        saw = ""
    if not isinstance(saw, str):
        fatal.append(f"line {lineno}: saw is not a string")
        saw = ""
    if not fatal:
        if len(want) != module_count:
            issues.append(f"width {len(want)} != module_count {module_count}")
        if not saw:
            issues.append("no saw baseline")
    return Record(index=index, photo=photo, want=want, saw=saw,
                  fatal=fatal, issues=issues)


def describe(ds: Dataset) -> dict:
    """Validation summary for the UI (counts, prefixes, flagged rows)."""
    runnable = ds.runnable
    widths = [r.width for r in ds.records if r.want]
    prefix_counts = Counter(prefix_of(r.photo) for r in runnable)
    flagged = [{"index": r.index, "photo": r.photo, "fatal": r.fatal,
                "issues": r.issues}
               for r in ds.records if r.fatal or r.issues]
    return {
        "path": ds.path,
        "exists": os.path.isfile(ds.path),
        "rows": len(ds.records),
        "runnable": len(runnable),
        "problems": list(ds.problems),
        "problems_count": len(ds.problems),
        "missing_photos": ds.missing_photos[:50],
        "missing_count": len(ds.missing_photos),
        "widths": {"min": min(widths), "max": max(widths)} if widths else None,
        "prefixes": [{"prefix": p, "rows": n}
                     for p, n in prefix_counts.most_common()],
        "flagged": flagged[:50],
        "flagged_count": len(flagged),
    }


def discover(filename: str = "reads.jsonl", limit: int = 30,
             tools_dir: str | None = None) -> list[str]:
    """Run dirs of the sibling tools that contain ``filename`` (newest first)."""
    if tools_dir is None:
        tools_dir = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
    found: list[str] = []
    for parts in _DISCOVER_ROOTS:
        runs_dir = os.path.join(tools_dir, *parts)
        try:
            entries = sorted(os.listdir(runs_dir), reverse=True)
        except OSError:
            continue
        for name in entries:
            candidate = os.path.join(runs_dir, name)
            if os.path.isfile(os.path.join(candidate, filename)):
                found.append(candidate)
            if len(found) >= limit:
                return found
    return found
