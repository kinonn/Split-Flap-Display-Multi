"""Scoring for the ocr-vlm benchmark (pure functions, no I/O).

The score of one row is the number of character positions where the read
differs from ``want`` — lower is better. The baseline (``saw``, produced
by a previous reader) is scored exactly the same way, and the run report
compares the two.

Normalization is applied to the VLM read, ``saw`` and ``want`` alike:

- uppercase every character (the drum charset is uppercase, so punishing
  a lowercase answer would be noise, not signal),
- map display aliases: ``␣``, ``·``, ``_`` and the words
  ``space``/``blank``/``empty``/``nothing`` to a space,
- strip pure decoration from around a value: backticks and double quotes
  are NOT drum characters. A lone apostrophe IS a drum character and is
  never stripped, and surrounding whitespace is never trimmed (blank
  flaps are real positions),
- pad short / truncate long values to the row width with spaces, flagging
  the row ``adjusted`` — chosen policy: adjusted rows are still counted,
  never silently dropped.
"""

from __future__ import annotations

from collections import Counter
from statistics import median

BLANK_WORDS = frozenset({"space", "blank", "empty", "nothing", "none"})

_ALIASES = {"\u2423": " ", "\u00b7": " ", "_": " "}

# Non-drum decoration around an answer (backtick, double quote).
_DECORATION = "`\""

# Quote characters a model might wrap ONE module answer in.
_WRAPPERS = "`'\""


def normalize_char(raw) -> str:
    """One module position: decoration off, aliases mapped, uppercased."""
    s = str(raw if raw is not None else "")
    if not s.strip():
        return " "
    s = s.strip()
    # Strip wrapping quotes around a single-answer string ("'A'" -> "A").
    # Safe here because a module can only ever hold one character, so a
    # leading and trailing quote cannot both be real readings.
    while len(s) >= 2 and s[0] == s[-1] and s[0] in _WRAPPERS:
        s = s[1:-1].strip()
    if not s:
        return " "
    key = s.lower()
    if key in BLANK_WORDS:
        return " "
    if key in _ALIASES:
        return _ALIASES[key]
    return s[0].upper()


def normalize_text(raw, width: int) -> tuple[str, bool]:
    """Whole value -> (text of exactly ``width`` chars, adjusted).

    ``adjusted`` is True when the length had to be changed by padding or
    truncation (the row is still scored, and flagged in the report).
    """
    s = str(raw if raw is not None else "")
    if s.strip().lower() in BLANK_WORDS:
        s = ""
    else:
        # Never strip whitespace: leading/trailing blanks are real flaps.
        s = s.strip(_DECORATION)
    chars = [normalize_char(ch) for ch in s]
    adjusted = len(chars) != width
    if len(chars) < width:
        chars.extend([" "] * (width - len(chars)))
    return "".join(chars[:width]), adjusted


def mismatches(got: str, want: str) -> int:
    """Positional character mismatches (both strings are width-aligned)."""
    return sum(1 for a, b in zip(got, want) if a != b)


def score_reading(read: str | None, want: str, saw: str, width: int) -> dict:
    """Normalize + compare one row's read and baseline against ``want``.

    ``read`` is the VLM answer (None when the read failed). Returns the
    comparable, JSON-serializable pieces of a row.
    """
    want_norm, want_adj = normalize_text(want, width)
    saw_norm: str | None = None
    saw_adj = False
    if saw:
        saw_norm, saw_adj = normalize_text(saw, width)
    read_norm: str | None = None
    read_adj = False
    if read is not None:
        read_norm, read_adj = normalize_text(read, width)
    return {
        "want_norm": want_norm,
        "saw_norm": saw_norm,
        "read_norm": read_norm,
        "mm_vlm": mismatches(read_norm, want_norm)
        if read_norm is not None else None,
        "mm_saw": mismatches(saw_norm, want_norm)
        if saw_norm is not None else None,
        "adjusted": bool(read_adj or saw_adj),
        "want_adjusted": want_adj,
    }


def _method_stats(values: list[int], scored_rows: list[dict], key: str,
                  width: int) -> dict:
    """Aggregates for one method (vlm or saw) over its scored rows."""
    n = len(values)
    hist_max = max([width, *values]) if values else width
    histogram = {str(k): 0 for k in range(hist_max + 1)}
    for v in values:
        histogram[str(v)] = histogram.get(str(v), 0) + 1
    pos_counts = [0] * width
    for r in scored_rows:
        got = r.get(key)
        want = r.get("want_norm") or ""
        if got is None:
            continue
        for i in range(width):
            if i < len(got) and i < len(want) and got[i] != want[i]:
                pos_counts[i] += 1
    exact = sum(1 for v in values if v == 0)
    return {
        "n": n,
        "total": sum(values) if values else None,
        "mean": round(sum(values) / n, 3) if n else None,
        "median": round(median(values), 3) if values else None,
        "exact": exact,
        "exact_pct": round(100.0 * exact / n, 1) if n else None,
        "histogram": histogram,
        "per_position": [round(100.0 * c / n, 1) if n else None
                         for c in pos_counts],
        "per_position_counts": pos_counts,
    }


def summarize(rows: list[dict], width: int) -> dict:
    """Aggregate stats for the run report and the UI summary card.

    Rows with ``error`` set (invalid line, photo unreadable, VLM failure)
    are excluded from every accuracy average and counted separately.
    """
    ok = [r for r in rows if not r.get("error")]
    vlm_rows = [r for r in ok if r.get("mm_vlm") is not None]
    saw_rows = [r for r in ok if r.get("mm_saw") is not None]
    vlm = _method_stats([r["mm_vlm"] for r in vlm_rows], vlm_rows,
                        "read_norm", width)
    saw = _method_stats([r["mm_saw"] for r in saw_rows], saw_rows,
                        "saw_norm", width)

    h2h = {"n": 0, "better": 0, "equal": 0, "worse": 0, "delta_mean": None}
    deltas: list[int] = []
    for r in ok:
        a, b = r.get("mm_vlm"), r.get("mm_saw")
        if a is None or b is None:
            continue
        h2h["n"] += 1
        if a < b:
            h2h["better"] += 1
        elif a > b:
            h2h["worse"] += 1
        else:
            h2h["equal"] += 1
        deltas.append(b - a)  # positive = VLM better than baseline
    if deltas:
        h2h["delta_mean"] = round(sum(deltas) / len(deltas), 3)

    conf: Counter[tuple[str, str]] = Counter()
    for r in vlm_rows:
        got, want = r.get("read_norm"), r.get("want_norm")
        if got is None or want is None:
            continue
        for i in range(min(len(got), len(want))):
            if got[i] != want[i]:
                conf[(want[i], got[i])] += 1
    confusions = [{"want": w, "got": g, "count": c}
                  for (w, g), c in conf.most_common(15)]

    groups: dict[str, list[dict]] = {}
    for r in ok:
        groups.setdefault(r.get("prefix") or "other", []).append(r)
    prefixes = []
    for name, rs in sorted(groups.items(),
                           key=lambda kv: (-len(kv[1]), kv[0])):
        vals = [r["mm_vlm"] for r in rs if r.get("mm_vlm") is not None]
        base = [r["mm_saw"] for r in rs if r.get("mm_saw") is not None]
        exact = sum(1 for v in vals if v == 0)
        prefixes.append({
            "prefix": name,
            "rows": len(rs),
            "vlm_avg": round(sum(vals) / len(vals), 2) if vals else None,
            "saw_avg": round(sum(base) / len(base), 2) if base else None,
            "vlm_exact": exact,
            "vlm_exact_pct": round(100.0 * exact / len(vals), 1)
            if vals else None,
        })

    return {
        "width": width,
        "rows": len(rows),
        "scored_vlm": len(vlm_rows),
        "scored_saw": len(saw_rows),
        "errors": sum(1 for r in rows if r.get("error")),
        "adjusted": sum(1 for r in rows if r.get("adjusted")),
        "vlm": vlm,
        "saw": saw,
        "head_to_head": h2h,
        "confusions": confusions,
        "prefixes": prefixes,
    }
