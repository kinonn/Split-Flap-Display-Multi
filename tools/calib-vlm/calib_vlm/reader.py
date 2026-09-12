"""VLM reader: one photo -> per-module reading, reconciled to the known
module count so leading/trailing blanks are never lost.

The model is asked for exactly one entry per module in left-to-right
order. When a provider still returns a compacted run (e.g. "AB" for a
display showing "  AB  "), the visible characters are aligned to the
known width against the expected frame's blank layout and every
inferred position is flagged, never silently trusted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import cv2

from .prompts import (READER_SYSTEM, REPORT_READING_TOOL, correction_text,
                      reader_user_text)
from .vlm import VLMError, image_part, text_part

CONDITIONS = ("clean", "half", "double", "blank", "unreadable")

_CONDITION_ALIASES = {
    "clean": "clean", "ok": "clean", "good": "clean", "clear": "clean",
    "perfect": "clean", "centered": "clean", "centred": "clean",
    "half": "half", "half-flap": "half", "half_flap": "half",
    "half flap": "half", "partial": "half", "split": "half",
    "double": "double", "double-flap": "double", "double_flap": "double",
    "double flap": "double", "two": "double", "overlap": "double",
    "blank": "blank", "empty": "blank", "space": "blank", "none": "blank",
    "unreadable": "unreadable", "unknown": "unreadable",
    "unclear": "unreadable", "": "unreadable",
}

_CHAR_ALIASES = {"\u2423": " ", "\u00b7": " ", "_": " ", "-": "-",
                 "space": " ", "blank": " ", "empty": " ", "nothing": " "}


class ReaderError(RuntimeError):
    pass


@dataclass
class ModuleReading:
    module: int
    char: str
    condition: str
    confidence: float
    source: str = "vlm"  # "vlm" | "inferred" | "error"
    expected: str = ""

    def as_dict(self) -> dict:
        return {"module": self.module, "char": self.char,
                "condition": self.condition,
                "confidence": round(self.confidence, 3),
                "source": self.source, "expected": self.expected}


@dataclass
class Reading:
    modules: list[ModuleReading]
    realigned: bool = False
    raw_count: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(m.char for m in self.modules)

    def as_dict(self) -> dict:
        return {"read": self.text, "realigned": self.realigned,
                "raw_count": self.raw_count, "warnings": self.warnings,
                "modules": [m.as_dict() for m in self.modules]}


def jpeg_bytes(img, max_width: int = 1024, quality: int = 80) -> bytes:
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ReaderError("JPEG encode failed")
    return bytes(buf)


def annotate_modules(img, total: int):
    """Draw faint module separators + index ticks so the model counts
    positions instead of trimming blanks. Returns a BGR image."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if total <= 1:
        return img
    h, w = img.shape[:2]
    top = 34 if total <= 24 else 24
    canvas = cv2.copyMakeBorder(img, top, 0, 0, 0, cv2.BORDER_CONSTANT,
                                value=(28, 28, 28))
    for i in range(1, total):
        x = int(round(w * i / total))
        cv2.line(canvas, (x, top), (x, top + h), (0, 150, 90), 1)
    scale = 0.42 if total <= 24 else 0.3
    for i in range(total):
        x0, x1 = int(round(w * i / total)), int(round(w * (i + 1) / total))
        cx = (x0 + x1) // 2
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(canvas, label, (max(0, cx - tw // 2), top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (220, 220, 220), 1,
                    cv2.LINE_AA)
    return canvas


def _normalize_char(raw, charset: str) -> str:
    s = str(raw if raw is not None else "").strip()
    low = s.lower()
    if low in _CHAR_ALIASES:
        return _CHAR_ALIASES[low]
    if s in _CHAR_ALIASES:
        return _CHAR_ALIASES[s]
    if not s:
        return " "
    allowed = set(charset)
    for ch in s:
        if ch in allowed:
            return ch
        if ch.upper() in allowed:  # models like to lowercase letters
            return ch.upper()
    return "?"


def _normalize_condition(raw) -> str:
    key = str(raw if raw is not None else "").strip().lower()
    return _CONDITION_ALIASES.get(key, "unreadable")


def _normalize_confidence(raw) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value > 1.0:  # some models answer on a 0..100 scale
        value = value / 100.0
    return max(0.0, min(1.0, value))


class VlmReader:
    """Turns one display photo into a width-reconciled per-module reading."""

    def __init__(self, vlm, annotate: bool = True):
        self.vlm = vlm
        self.annotate = annotate

    # -- message assembly -----------------------------------------------------
    def _messages(self, total: int, charset: str, drum: str, jpeg: bytes,
                  extra: str | None = None) -> list[dict]:
        parts = [text_part(reader_user_text(total, charset, drum)),
                 image_part(jpeg)]
        if extra:
            parts.append(text_part(extra))
        return [{"role": "system", "content": READER_SYSTEM},
                {"role": "user", "content": parts}]

    def _extract(self, reply: dict) -> list[tuple[str, str, float]] | None:
        """Pull the modules array out of the tool call; None when absent."""
        for call in reply.get("tool_calls") or []:
            if call.get("name") != "report_reading":
                continue
            args = call.get("arguments") or {}
            if "_parse_error" in args:
                return None
            raw = args.get("modules")
            if not isinstance(raw, list) or not raw:
                return None
            entries = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                entries.append((item.get("char"), item.get("condition"),
                                item.get("confidence")))
            return entries or None
        return None

    @staticmethod
    def _content_entries(reply: dict) -> list[tuple[str, str, float]] | None:
        """Prose fallback: a JSON {"modules": [...]} object in the reply
        content is as good as a tool call.

        Needed because thinking-mode providers ignore forced tool_choice
        (they answer in content); without this every such frame would be
        an unreadable degraded read instead of a real reading.
        """
        content = reply.get("content")
        if not isinstance(content, str):
            return None
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
        try:
            parsed = json.loads(text)
        except ValueError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return None
            try:
                parsed = json.loads(match.group(0))
            except ValueError:
                return None
        if not isinstance(parsed, dict):
            return None
        raw = parsed.get("modules")
        if not isinstance(raw, list) or not raw:
            return None
        entries = []
        for item in raw:
            if isinstance(item, dict):
                entries.append((item.get("char"), item.get("condition"),
                                item.get("confidence")))
        return entries or None

    # -- public ---------------------------------------------------------------
    def read(self, jpeg: bytes, total: int, expected: str = "",
             charset: str = "", drum: str = "") -> Reading:
        """One photo -> reconciled reading.

        Shorter answers are the collapsed-blanks case and are reconciled
        against the expected blank layout immediately (flagged). An answer
        longer than the display is a schema violation and gets a single
        corrective retry, then is reconciled/flagged rather than trusted.
        """
        last_problem = "model did not call report_reading"
        normalized: list[tuple[str, str, float]] | None = None
        got = 0
        # Actual round trips of the last read() call (a re-ask counts too);
        # the calibrator charges these against its VLM call budget.
        self.last_calls = 0
        for attempt in range(2):
            extra = None if attempt == 0 else correction_text(total, got)
            messages = self._messages(total, charset, drum, jpeg, extra)
            self.last_calls += 1
            try:
                reply = self.vlm.chat(messages, tools=[REPORT_READING_TOOL],
                                      tool_choice="required")
            except VLMError as exc:
                raise ReaderError(str(exc)) from exc
            entries = self._extract(reply)
            if entries is None:
                # Prose fallback: with tool_choice dropped, thinking
                # models often answer as JSON content instead of a call.
                entries = self._content_entries(reply)
            if entries is None:
                last_problem = "model did not call report_reading"
                continue
            normalized = [(_normalize_char(ch, charset),
                           _normalize_condition(cond),
                           _normalize_confidence(conf))
                          for ch, cond, conf in entries]
            got = len(normalized)
            if got <= total:
                # Shorter than the display is the collapsed-blanks case:
                # reconcile immediately (and loudly flag it). Longer than
                # the display is a real schema violation -> retry.
                return self._reconcile(normalized, total, expected,
                                       raw_count=got)
            last_problem = f"model returned {got} entries for {total} modules"
        if normalized is None:
            raise ReaderError(last_problem)
        return self._reconcile(normalized, total, expected, raw_count=got)

    def error_reading(self, total: int, expected: str, problem: str) -> Reading:
        """All-unreadable stand-in so a failed read degrades gracefully."""
        modules = [ModuleReading(i, "?", "unreadable", 0.0, "error",
                                 expected[i] if i < len(expected) else " ")
                   for i in range(total)]
        return Reading(modules, realigned=False, raw_count=0,
                       warnings=[problem])

    # -- reconciliation -------------------------------------------------------
    @staticmethod
    def _best_fit(entries: list[tuple[str, str, float]], expected: str,
                  total: int) -> int:
        """Offset placing the visible run on the best expected positions.

        Prefers positions the expected frame says are non-blank (a blank
        flap is visually indistinguishable, so a visible glyph belongs
        where a real character was commanded), then exact matches, then
        the leftmost offset.
        """
        m = len(entries)
        best = (0, 0, 0)  # (on_nonblank, matches, -offset)
        for offset in range(0, max(0, total - m) + 1):
            on_nonblank = 0
            matches = 0
            for j, (ch, _cond, _conf) in enumerate(entries):
                pos = offset + j
                exp = expected[pos] if pos < len(expected) else " "
                if exp != " ":
                    on_nonblank += 1
                if ch == exp:
                    matches += 1
            score = (on_nonblank, matches, -offset)
            if score > best:
                best = score
        return -best[2]

    def _reconcile(self, entries: list[tuple[str, str, float]], total: int,
                   expected: str, raw_count: int) -> Reading:
        warnings: list[str] = []
        realigned = False
        if len(entries) == total:
            ordered: list[tuple[str, str, float] | None] = list(entries)
        elif len(entries) < total:
            offset = (self._best_fit(entries, expected, total)
                      if expected else 0)
            ordered = [None] * total
            for j, entry in enumerate(entries):
                if offset + j < total:
                    ordered[offset + j] = entry
            realigned = True
            warnings.append(
                f"model returned {len(entries)}/{total} entries; visible run "
                f"aligned at offset {offset} and blanks inferred from the "
                "commanded frame")
        else:
            ordered = list(entries[:total])
            realigned = True
            warnings.append(f"model returned {len(entries)}/{total} entries; "
                            "extra entries dropped")
        modules: list[ModuleReading] = []
        for pos in range(total):
            exp = expected[pos] if pos < len(expected) else " "
            entry = ordered[pos] if pos < len(ordered) else None
            if entry is None:
                if exp == " ":
                    modules.append(ModuleReading(pos, " ", "blank", 0.0,
                                                 "inferred", exp))
                else:
                    modules.append(ModuleReading(pos, "?", "unreadable", 0.0,
                                                 "inferred", exp))
                continue
            ch, cond, conf = entry
            if ch == " " and exp != " ":
                # A visible glyph was commanded but the model says blank:
                # keep the reading, it is a real failure signal.
                cond = "blank"
            modules.append(ModuleReading(pos, ch, cond, conf, "vlm", exp))
        return Reading(modules, realigned=realigned, raw_count=raw_count,
                       warnings=warnings)
