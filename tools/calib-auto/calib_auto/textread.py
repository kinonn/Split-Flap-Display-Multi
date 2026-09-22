"""OCR text read path and read-mode dispatch (benchmark side).

Some vision models (PaddleOCR-style OCR specialists) cannot make function
calls, so the tool reader fails on every frame with "model did not call
report_reading". This module adds a second read path that sends the
photo with a plain OCR prompt and parses the reply as characters, plus a
dispatcher that picks between the two:

- ``"tool"``: the per-module ``report_reading`` tool call (the reader
  the calibration loop uses, and the one that produced the ``saw``
  baselines of the historical datasets).
- ``"text"``: native OCR prompt (default ``"OCR:"`` — probed against a
  PaddleOCR-VL server, which ignores instruction-style prompts) + text
  parse.
- ``"auto"``: probe with the tool reader on the first read of a run; if
  it raises ``ReaderError``, switch that reader permanently to the text
  path for the rest of the run.

The text parser handles the observed reply shapes: space-separated
characters (``% % % % # %``), contiguous runs (``ABCDEFGHIJKL``) and
merged chunks (``MMMJN% % BBCN``). Whitespace runs are concatenated as
characters, blank words ("space") count as one blank module, and a reply
that is shorter/longer than the display width is padded/truncated and
flagged — the same policy as the tool reader, never a silently dropped
row. Positional blanks that an OCR text output does not encode are
therefore lost: the row keeps the raw reply and reads worse, honestly.

Detected modes (``montage``, ``cells-detect``, ``strip-detect``) first
locate the display with OpenCV (``calib_auto/segment.py``) and use the
real module grid, so per-module reads are position-exact, per-cell crops
are centred, blanks are decided locally (no model call, no blank
hallucination), and the image can be normalized before it is sent. When
the display is not detected the read falls back to the plain strip path
and the row is flagged ``no-detect``.
"""

from __future__ import annotations

import base64
import json
import re

import cv2

from .reader import (
    ModuleReading,
    ReaderError,
    Reading,
    VlmReader,
    _normalize_char,
    annotate_modules,
    jpeg_bytes,
)
from .vlm import VLMError, text_part

DEFAULT_OCR_PROMPT = "OCR:"

# Cell (per-glyph) mode: trim this fraction off each side of a module crop
# so a sliver of the neighbouring flap cannot bleed into the image.
CELL_INSET = 0.02

# Words a model may write where a blank flap sits.
_BLANK_WORDS = frozenset({"space", "blank", "empty", "nothing", "none"})

# Decoration around a token. NOT the apostrophe: it is a real drum
# character, so "'" must survive token cleanup.
_TOKEN_WRAPPERS = "`\""

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def _chars_from_json(text: str, charset: str) -> list[str] | None:
    """Characters from a JSON reply (models that answer in JSON anyway).

    Accepts ``{"modules": [...]}`` (the tool schema, in case a
    model replies in prose JSON), ``{"chars"|"characters"|"text": ...}``
    and a bare array of strings/objects.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE_RE.sub("", stripped).strip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        parsed = json.loads(stripped)
    except ValueError:
        return None
    items = None
    if isinstance(parsed, dict):
        for key in ("modules", "chars", "characters", "text", "result"):
            value = parsed.get(key)
            if isinstance(value, (list, str)):
                items = value
                break
    elif isinstance(parsed, list):
        items = parsed
    if items is None:
        return None
    if isinstance(items, str):
        items = [items]
    chars: list[str] = []
    for item in items:
        if isinstance(item, dict):
            item = item.get("char", item.get("text", ""))
        for ch in str(item):
            chars.append(_normalize_char(ch, charset))
    return chars


def parse_ocr_text(content, width: int, charset: str) -> tuple[list[str], int, list[str]]:
    """Reply text -> (chars of exactly ``width``, raw length, warnings)."""
    warnings: list[str] = []
    text = "" if content is None else str(content)
    chars = _chars_from_json(text, charset)
    if chars is None:
        chars = []
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = _FENCE_RE.sub("", stripped).strip()
        elif (len(stripped) >= 2 and stripped[0] == stripped[-1]
                and stripped[0] in _TOKEN_WRAPPERS):
            stripped = stripped[1:-1]
        for token in stripped.split():
            token = token.strip(_TOKEN_WRAPPERS)
            if not token:
                continue
            if token.lower() in _BLANK_WORDS:
                chars.append(" ")
                continue
            for ch in token:
                chars.append(_normalize_char(ch, charset))
    raw_len = len(chars)
    if raw_len == 0:
        warnings.append(
            "empty reply: no characters detected (read as all blanks)")
    if raw_len < width:
        chars = chars + [" "] * (width - raw_len)
        warnings.append(
            f"reply had {raw_len} of {width} characters; padded with blanks")
    elif raw_len > width:
        warnings.append(
            f"reply had {raw_len} characters for {width} modules; truncated")
        chars = chars[:width]
    return chars, raw_len, warnings


def cell_crops(image, total: int, inset: float = CELL_INSET):
    """Split a display photo into per-module crops, left to right.

    The module grid is the same convention the tool-call path annotates:
    module ``i`` spans ``[w*i/total, w*(i+1)/total)``. Reading one crop per
    request gives every glyph the whole image budget of the vision encoder
    and removes the sequence/counting problem of reading a long thin strip
    (merged runs, dropped blanks, miscounted positions).
    """
    width = image.shape[1]
    for i in range(max(1, total)):
        x0 = round(width * i / total)
        x1 = round(width * (i + 1) / total)
        margin = round((x1 - x0) * max(0.0, min(0.2, inset)))
        if x1 - margin > x0 + margin:
            x0, x1 = x0 + margin, x1 - margin
        yield image[:, x0:x1]


def extract_single_glyph(content, charset: str) -> tuple[str, bool]:
    """Reply for a ONE-glyph cell image -> (character, reply was empty).

    Accepts the common shapes a model returns for a single-glyph crop:
    a bare character, a decorated one (``**M**``), a short phrase
    ("The character is M" — the first standalone one-character token
    wins) and blank words ("space"). An empty reply means a blank cell.
    """
    text = "" if content is None else str(content)
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE_RE.sub("", stripped).strip()
    if not stripped:
        return " ", True
    plain = stripped.strip(_TOKEN_WRAPPERS)
    if len(plain) == 1:
        return _normalize_char(plain, charset), False
    for token in stripped.split():
        bare = token.strip(_TOKEN_WRAPPERS + "*")
        if bare.lower() in _BLANK_WORDS:
            return " ", False
        if len(bare) == 1:
            ch = _normalize_char(bare, charset)
            if ch != " ":
                return ch, False
    # Last resort: the general parser's first character of the reply.
    chars, raw_len, _ = parse_ocr_text(content, 1, charset)
    return chars[0], raw_len == 0


class TextReader:
    """One photo -> Reading via an OCR prompt (no function calling).

    The photo is encoded here (not upstream) so the image settings —
    ``max_width``, JPEG ``quality``, ``fmt`` — apply to this path only;
    the tool-call path keeps its production-parity encoding.
    """

    def __init__(self, vlm, prompt: str = DEFAULT_OCR_PROMPT,
                 max_width: int = 1024, quality: int = 80,
                 fmt: str = "jpeg", max_tokens: int | None = None,
                 image_mode: str = "strip", preprocess: str = "none",
                 blank_gate: bool = False):
        self.vlm = vlm
        self.prompt = prompt or DEFAULT_OCR_PROMPT
        self.max_width = max(64, int(max_width))
        self.quality = min(100, max(1, int(quality)))
        self.fmt = "png" if str(fmt).lower() == "png" else "jpeg"
        # "strip" = the whole photo in one request; "cells" = one glyph per
        # module per request (exact position mapping, 12x the calls);
        # "montage"/"cells-detect"/"strip-detect" = OpenCV-located display
        # grid with per-module crops (see calib_auto/segment.py).
        from . import segment

        mode = str(image_mode).lower()
        self.image_mode = mode if mode in (
            "strip", "cells", *segment.DETECTED_MODES) else "strip"
        self.preprocess = (str(preprocess).lower()
                           if str(preprocess).lower()
                           in segment.PREPROCESS_STYLES else "none")
        # Opt-in OpenCV blank gate: when off (default) every cell goes to
        # the model and no cell is short-circuited or corrected to blank.
        self.blank_gate = bool(blank_gate)
        # Cap generation (None = provider default). An OCR answer needs a
        # handful of tokens; uncapped, a model that loops on a row of one
        # repeated character runs to the server's default limit for output
        # the parser never looks at.
        self.max_tokens = int(max_tokens) if max_tokens else None
        self.last_calls = 0
        self.last_raw: str | None = None
        self.last_empty = False
        # Detected-mode metadata, read by the server after every read():
        # whether a display was found, its box, per-cell blank decisions
        # and the composed image that was (or would be) sent.
        self.last_no_detect = False
        self.last_detected: bool | None = None
        self.last_display: dict | None = None
        self.last_blanks: list[bool] | None = None
        self.last_composed = None

    def encode(self, image) -> tuple[bytes, str]:
        """BGR photo -> (bytes, media type) honouring the image settings."""
        if self.fmt == "png":
            h, w = image.shape[:2]
            if w > self.max_width:
                image = cv2.resize(image, (self.max_width,
                                           int(h * self.max_width / w)))
            ok, buf = cv2.imencode(".png", image)
            if not ok:
                raise ReaderError("PNG encode failed")
            return bytes(buf), "image/png"
        return (jpeg_bytes(image, max_width=self.max_width,
                           quality=self.quality), "image/jpeg")

    @staticmethod
    def _image_part(data: bytes, media: str) -> dict:
        return {"type": "image_url",
                "image_url": {"url": f"data:{media};base64,"
                                       + base64.b64encode(data).decode()}}

    def read(self, image, total: int, expected: str = "",
             charset: str = "", drum: str = "") -> Reading:
        from . import segment

        if self.image_mode == "cells":
            return self._read_cells(image, total, expected, charset)
        if self.image_mode in segment.DETECTED_MODES:
            return self._read_detected(image, total, expected, charset)
        return self._read_strip_from(image, total, expected, charset)

    def _read_strip_from(self, image, total: int, expected: str,
                         charset: str) -> Reading:
        data, media = self.encode(image)
        messages = [{"role": "user", "content": [
            text_part(self.prompt), self._image_part(data, media)]}]
        self.last_calls += 1
        try:
            reply = self.vlm.chat(messages, max_tokens=self.max_tokens)
        except VLMError as exc:
            raise ReaderError(str(exc)) from exc
        content = reply.get("content")
        self.last_raw = content if isinstance(content, str) else None
        chars, raw_len, warnings = parse_ocr_text(content, total, charset)
        self.last_empty = raw_len == 0
        modules = [
            ModuleReading(i, ch, "blank" if ch == " " else "clean",
                          1.0, "text",
                          expected[i] if i < len(expected) else " ")
            for i, ch in enumerate(chars[:total])
        ]
        return Reading(modules, realigned=raw_len != total,
                       raw_count=raw_len, warnings=warnings)

    # -- OpenCV-detected modes ------------------------------------------------
    def _read_detected(self, image, total: int, expected: str,
                       charset: str) -> Reading:
        """Locate the display, crop modules, then read per the mode.

        Detection failure is not fatal: the photo is read as a plain
        strip (the pre-segmentation behavior) and the caller flags the
        row, so a bad detector degrades to the old numbers instead of
        cropping garbage.
        """
        from . import segment

        self.last_no_detect = False
        self.last_detected = None
        self.last_display = None
        self.last_blanks = None
        self.last_composed = None
        display = segment.find_display(image)
        if display is None:
            self.last_no_detect = True
            self.last_detected = False
            reading = self._read_strip_from(image, total, expected, charset)
            reading.warnings.insert(
                0, "display not detected; read the whole photo as strip")
            return reading
        boxes = segment.module_boxes(image, display, total)
        crops = segment.crop_modules(image, boxes)
        blanks = ([segment.is_blank(crop) for crop in crops]
                  if self.blank_gate else [False] * len(crops))
        self.last_detected = True
        self.last_display = display.as_dict()
        self.last_blanks = blanks
        if self.image_mode == "cells-detect":
            return self._read_cells_detected(crops, blanks, total, expected,
                                             charset)
        if self.image_mode == "strip-detect":
            return self._read_strip_detected(image, display, total, expected,
                                             charset)
        return self._read_montage(crops, blanks, total, expected, charset)

    def _read_strip_detected(self, image, display, total: int, expected: str,
                             charset: str) -> Reading:
        from . import segment

        h, w = image.shape[:2]
        x0 = max(0, int(display.x0))
        x1 = min(w, int(display.x1))
        y0 = max(0, display.y0)
        y1 = min(h, display.y1 + 1)
        styled = segment.apply_style(image[y0:y1, x0:x1], self.preprocess)
        self.last_composed = cv2.cvtColor(styled, cv2.COLOR_GRAY2BGR)
        return self._read_strip_from(self.last_composed, total, expected,
                                     charset)

    def _read_cells_detected(self, crops, blanks, total: int, expected: str,
                             charset: str) -> Reading:
        """One request per module (opt-in: per non-blank module).

        With ``blank_gate`` on, blank cells never reach the model — the
        failure that sank the old cells mode was hallucinated output for
        isolated blank crops. With it off (default) every cell is sent.
        """
        from . import segment

        chars: list[str] = []
        raws: list[str] = []
        skipped = 0
        for crop, blank in zip(crops, blanks):
            if blank:
                chars.append(" ")
                raws.append("[blank]")
                skipped += 1
                continue
            data, media = self.encode(
                segment.normalize(crop, self.preprocess))
            messages = [{"role": "user", "content": [
                text_part(self.prompt), self._image_part(data, media)]}]
            self.last_calls += 1
            try:
                reply = self.vlm.chat(messages, max_tokens=self.max_tokens)
            except VLMError as exc:
                raise ReaderError(str(exc)) from exc
            content = reply.get("content")
            raws.append("" if content is None else str(content).strip())
            ch, _empty = extract_single_glyph(content, charset)
            chars.append(ch)
        shown = "|".join(raw.replace("\n", "\\n") for raw in raws)
        self.last_raw = shown[:500]
        self.last_empty = all(ch == " " for ch in chars)
        self.last_composed = segment.montage(crops, total,
                                             style=self.preprocess)
        warnings: list[str] = []
        if skipped and self.blank_gate:
            warnings.append(f"{skipped}/{total} blank cells decided by "
                            "OpenCV (no model call)")
        modules = [
            ModuleReading(i, ch, "blank" if ch == " " else "clean", 1.0,
                          "cv" if i < len(blanks) and blanks[i] else "text",
                          expected[i] if i < len(expected) else " ")
            for i, ch in enumerate(chars[:total])
        ]
        return Reading(modules, realigned=False,
                       raw_count=sum(1 for ch in chars if ch != " "),
                       warnings=warnings)

    def _read_montage(self, crops, blanks, total: int, expected: str,
                      charset: str) -> Reading:
        """One labeled contact sheet of all modules per request.

        Positions come from the grid layout, so a reply maps to modules
        by index. A reply that needed padding/truncation is not
        position-trustworthy and is kept as parsed (flagged), never
        overridden by the local blank decisions.
        """
        from . import segment

        sheet = segment.montage(crops, total, style=self.preprocess)
        self.last_composed = sheet
        data, media = self.encode(sheet)
        messages = [{"role": "user", "content": [
            text_part(self.prompt), self._image_part(data, media)]}]
        self.last_calls += 1
        try:
            reply = self.vlm.chat(messages, max_tokens=self.max_tokens)
        except VLMError as exc:
            raise ReaderError(str(exc)) from exc
        content = reply.get("content")
        self.last_raw = content if isinstance(content, str) else None
        chars, raw_len, warnings = parse_ocr_text(content, total, charset)
        self.last_empty = raw_len == 0
        corrected = 0
        if self.blank_gate and raw_len == total:
            for i, blank in enumerate(blanks):
                if blank and chars[i] != " ":
                    chars[i] = " "
                    corrected += 1
        if corrected:
            warnings.append(f"{corrected} blank cells corrected by OpenCV "
                            "(cell content ignored)")
        modules = [
            ModuleReading(i, ch, "blank" if ch == " " else "clean", 1.0,
                          "cv" if i < len(blanks) and blanks[i] else "text",
                          expected[i] if i < len(expected) else " ")
            for i, ch in enumerate(chars[:total])
        ]
        return Reading(modules, realigned=raw_len != total,
                       raw_count=raw_len, warnings=warnings)

    def _read_cells(self, image, total: int, expected: str,
                    charset: str) -> Reading:
        """Per-glyph segmentation: one module crop per request.

        Each reply belongs to exactly one known module, so positions can
        never drift and an empty reply is simply that cell's blank. The
        bill is `total` requests per photo instead of one.
        """
        chars: list[str] = []
        raws: list[str] = []
        empties = 0
        for crop in cell_crops(image, total):
            data, media = self.encode(crop)
            messages = [{"role": "user", "content": [
                text_part(self.prompt), self._image_part(data, media)]}]
            self.last_calls += 1
            try:
                reply = self.vlm.chat(messages, max_tokens=self.max_tokens)
            except VLMError as exc:
                raise ReaderError(str(exc)) from exc
            content = reply.get("content")
            raws.append("" if content is None else str(content).strip())
            ch, empty = extract_single_glyph(content, charset)
            chars.append(ch)
            empties += 1 if empty else 0
        shown = "|".join(raw.replace("\n", "\\n") for raw in raws)
        self.last_raw = shown[:500]
        self.last_empty = empties == total
        warnings: list[str] = []
        if empties:
            warnings.append(f"{empties}/{total} cells returned an empty reply "
                            "(read as blanks)")
        modules = [
            ModuleReading(i, ch, "blank" if ch == " " else "clean",
                          1.0, "text",
                          expected[i] if i < len(expected) else " ")
            for i, ch in enumerate(chars[:total])
        ]
        return Reading(modules, realigned=False,
                       raw_count=sum(1 for ch in chars if ch != " "),
                       warnings=warnings)


class ModeReader:
    """Dispatch between the tool reader and the text reader.

    ``auto`` probes once per reader (i.e. per worker thread): the first
    read goes to the tool reader, and only if it raises ``ReaderError``
    does the reader switch to the text path for good. A later failure
    after a successful tool read is a genuine read failure (error row),
    not a mode switch — the benchmark stays consistent within a run.
    """

    def __init__(self, vlm, mode: str = "auto",
                 ocr_prompt: str = DEFAULT_OCR_PROMPT,
                 annotate: bool = True,
                 image_max_width: int = 1024, image_quality: int = 80,
                 image_format: str = "jpeg",
                 ocr_max_tokens: int | None = None,
                 image_mode: str = "strip", preprocess: str = "none",
                 blank_gate: bool = False):
        self.vlm = vlm
        self.mode = mode if mode in ("auto", "tool", "text") else "auto"
        self.tool = VlmReader(vlm, annotate=annotate)
        self.text = TextReader(vlm, ocr_prompt, max_width=image_max_width,
                               quality=image_quality, fmt=image_format,
                               max_tokens=ocr_max_tokens,
                               image_mode=image_mode,
                               preprocess=preprocess,
                               blank_gate=blank_gate)
        self._effective = "text" if self.mode == "text" else "tool"
        self._reads = 0
        # Row metadata, read by the server after every read() call.
        self.last_mode = self._effective
        self.last_fallback = False
        self.last_empty = False
        self.last_raw: str | None = None
        self.last_no_detect = False
        self.last_detected: bool | None = None
        self.last_display: dict | None = None
        self.last_blanks: list[bool] | None = None
        self.last_composed = None

    @property
    def annotate(self) -> bool:
        return self.tool.annotate

    def read(self, image, total: int, expected: str = "",
             charset: str = "", drum: str = "") -> Reading:
        """``image`` is the raw BGR photo (what ``cv2.imread`` returns).

        The two paths need different image preparation: the tool reader
        gets calib-vlm's module annotation (separators + index ticks),
        exactly like the production reads that produced ``saw``; the OCR
        text path must get the untouched photo — a probed PaddleOCR-VL
        server reads the annotation's index digits as content
        ("0 1 2 ... 11") and misses the display underneath. The text
        path also honours the configured image settings (max width,
        JPEG quality, PNG/JPEG format).
        """
        probe = self.mode == "auto" and self._reads == 0
        self._reads += 1
        self.last_fallback = False
        self.last_empty = False
        self.last_raw = None
        self.last_no_detect = False
        self.last_detected = None
        self.last_display = None
        self.last_blanks = None
        self.last_composed = None
        if self._effective == "tool":
            try:
                reading = self.tool.read(self._tool_jpeg(image, total), total,
                                         expected=expected, charset=charset,
                                         drum=drum)
            except ReaderError as exc:
                if not probe:
                    raise
                self._effective = "text"
                return self._text_read(image, total, expected, charset,
                                       fallback_reason=str(exc))
            self.last_mode = "tool"
            return reading
        return self._text_read(image, total, expected, charset)

    def _tool_jpeg(self, image, total: int) -> bytes:
        if self.tool.annotate:
            return jpeg_bytes(annotate_modules(image, total))
        return jpeg_bytes(image)

    def _text_read(self, image, total: int, expected: str, charset: str,
                   fallback_reason: str | None = None) -> Reading:
        reading = self.text.read(image, total, expected=expected,
                                 charset=charset)
        if fallback_reason:
            self.last_fallback = True
            reading.warnings.insert(
                0, f"tool-call read unavailable ({fallback_reason}); "
                   "used OCR text mode")
        self.last_mode = "text"
        self.last_raw = self.text.last_raw
        self.last_empty = self.text.last_empty
        self.last_no_detect = bool(getattr(self.text, "last_no_detect", False))
        self.last_detected = getattr(self.text, "last_detected", None)
        self.last_display = getattr(self.text, "last_display", None)
        self.last_blanks = getattr(self.text, "last_blanks", None)
        self.last_composed = getattr(self.text, "last_composed", None)
        return reading
