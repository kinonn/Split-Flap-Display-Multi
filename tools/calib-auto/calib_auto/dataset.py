"""Dataset loading and validation for the benchmark.

A dataset is a JSONL file whose lines look like::

    {"photo": "sw_37_f255.png", "want": "%%%%%%%%%%%%", "saw": "%%%%%%%#%%%%"}

``photo`` names an image in the SAME directory as the JSONL file, ``want``
is the frame that was commanded when the photo was taken and ``saw`` is
what the reader under test (the baseline) reported back then.

A *golden set* (see ``calib_auto/golden.py``) is the corrected variant:
``labels.jsonl`` lines carry the human-verified ``content`` as truth and
the previous reader's ``prior_read`` as the baseline, photos live in an
``images/`` subfolder, and every entry has a ``status``. Pending
(unverified) entries are excluded by default so pre-filled rows are
never mistaken for ground truth. Everything in this module is
read-only: it parses, validates and resolves paths and never writes.
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

# A golden set's manifest (see calib_auto/golden.py). ``baseline.jsonl``
# is the pre-rename name sets imported from older tools may carry.
GOLDEN_FILE = "labels.jsonl"
LEGACY_FILES = ("baseline.jsonl",)


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
    kind: str = "reads"                # "reads" | "golden"
    verified: int = 0                  # golden sets: entry status counts
    pending: int = 0
    skipped_pending: int = 0           # pending entries excluded from records

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
    """Resolve a photo inside the dataset directory (None when missing).

    Golden sets keep their copies in an ``images/`` subfolder, so the
    nested location is tried when the name is not beside the manifest.
    """
    if not valid_photo_name(name):
        return None
    path = os.path.join(directory, name)
    if os.path.isfile(path):
        return path
    nested = os.path.join(directory, "images", name)
    return nested if os.path.isfile(nested) else None


def prefix_of(photo: str) -> str:
    """Tag family of a photo name (``sw_37_f255.png`` -> ``sw``).

    The dataset groups run in tagged families (``sw_``, ``p0_``,
    ``ladder_``, ...); the raw prefix is the part before the first
    underscore, which is exactly the family for every known tag.
    """
    stem = os.path.splitext(os.path.basename(photo))[0]
    return stem.split("_")[0] or "other"


def resolve_dataset_file(directory: str, filename: str = GOLDEN_FILE) -> str:
    """The manifest that exists: ``filename``, else a legacy name.

    The fallback applies to the *default* name only: pointing at a set
    folder with ``labels.jsonl`` selected just works even when the set
    was imported before the rename (``baseline.jsonl``), while a
    deliberately typed different name is honoured (a typo then surfaces
    in Validate instead of silently loading something else).
    """
    directory = str(directory or "")
    filename = str(filename or GOLDEN_FILE)
    if os.path.isfile(os.path.join(directory, filename)):
        return filename
    if filename == GOLDEN_FILE:
        for legacy in LEGACY_FILES:
            if os.path.isfile(os.path.join(directory, legacy)):
                return legacy
    return filename


def load_dataset(directory: str, filename: str = GOLDEN_FILE,
                 module_count: int = 12,
                 verified_only: bool = True) -> Dataset:
    """Parse and validate a dataset file (photos are checked on disk).

    Handles both formats: a golden set's ``labels.jsonl`` (``content``
    = human truth, ``prior_read`` = baseline) and the legacy
    ``reads.jsonl`` (``want`` + ``saw``). For golden sets
    ``verified_only`` keeps pending entries out of ``records`` (counted
    in ``skipped_pending``); with it off they load but are flagged.

    Malformed lines become dataset-level ``problems``. Missing fields,
    bad widths and missing photo files become per-record ``fatal``/
    ``issues`` markers so every line shows up in the UI with a reason
    instead of being silently dropped.
    """
    directory = str(directory or "")
    filename = resolve_dataset_file(directory, filename)
    ds = Dataset(directory=directory, filename=filename)
    ds.kind = "golden" if filename in (GOLDEN_FILE, *LEGACY_FILES) else "reads"
    golden = ds.kind == "golden"
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
        status = str(item.get("status") or "pending") if golden else ""
        if golden:
            if status == "verified":
                ds.verified += 1
            else:
                ds.pending += 1
            if status != "verified" and verified_only:
                ds.skipped_pending += 1
                continue
        rec = _record_from_item(len(ds.records), lineno, item, module_count,
                                golden=golden)
        if golden and status != "verified":
            rec.issues.append("unverified (pending curation)")
        if rec.photo in seen:
            rec.issues.append(
                f"{ISSUE_DUPLICATE} (first at line {seen[rec.photo]})")
        else:
            seen[rec.photo] = lineno
        if (rec.runnable and rec.photo
                and photo_path(directory, rec.photo) is None):
            rec.issues.append(ISSUE_PHOTO)
            if rec.photo not in ds.missing_photos:
                ds.missing_photos.append(rec.photo)
        ds.records.append(rec)
    return ds


def _record_from_item(index: int, lineno: int, item: dict,
                      module_count: int, golden: bool = False) -> Record:
    photo = item.get("photo")
    # Golden entries keep the human truth in ``content`` and the
    # previous reader's answer in ``prior_read``; reads.jsonl uses the
    # ``want``/``saw`` pair. Normalize both into the Record fields.
    want = item.get("content") if golden else item.get("want")
    saw = item.get("prior_read", "") if golden else item.get("saw", "")
    kind_word = "content" if golden else "want"
    fatal: list[str] = []
    issues: list[str] = []
    if not isinstance(photo, str) or not photo.strip():
        fatal.append(f"line {lineno}: missing photo name")
        photo = ""
    elif not valid_photo_name(photo):
        fatal.append(f"line {lineno}: bad photo name {photo!r}")
    if not isinstance(want, str) or not want:
        fatal.append(f"line {lineno}: missing {kind_word} string")
        want = ""
    if saw is None:
        saw = ""
    if not isinstance(saw, str):
        fatal.append(f"line {lineno}: baseline read is not a string")
        saw = ""
    if not fatal:
        if len(want) != module_count:
            issues.append(f"width {len(want)} != module_count {module_count}")
        if not saw and not golden:
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
        "kind": ds.kind,
        "verified": ds.verified,
        "pending": ds.pending,
        "skipped_pending": ds.skipped_pending,
        "rows": len(ds.records),
        "runnable": len(runnable),
        "problems": list(ds.problems),
        "problems_count": len(ds.problems),
        "missing_photos": ds.missing_photos[:50],
        "missing_count": len(ds.missing_photos),
        "widths": {"min": min(widths), "max": max(widths)} if widths else None,
        "prefixes": [{"prefix": p, "rows": n}
                     for p, n in prefix_counts.most_common()],
        "sample_photos": [r.photo for r in runnable[:20]],
        "flagged": flagged[:50],
        "flagged_count": len(flagged),
    }
