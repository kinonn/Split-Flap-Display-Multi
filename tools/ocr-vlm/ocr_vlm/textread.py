"""OCR text read path and read-mode dispatch.

Some vision models (PaddleOCR-style OCR specialists) cannot make function
calls, so the calib-vlm tool reader fails on every frame with "model did
not call report_reading". This module adds a second read path that sends
the photo with a plain OCR prompt and parses the reply as characters,
plus a dispatcher that picks between the two:

- ``"tool"``: calib-vlm's per-module ``report_reading`` tool call (the
  reader that produced the ``saw`` baseline).
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
"""

from __future__ import annotations

import base64
import json
import re

import cv2

from calib_vlm.reader import (ModuleReading, ReaderError, Reading, VlmReader,
                              _normalize_char, annotate_modules,
                              jpeg_bytes)
from calib_vlm.vlm import VLMError, text_part

DEFAULT_OCR_PROMPT = "OCR:"

# Words a model may write where a blank flap sits.
_BLANK_WORDS = frozenset({"space", "blank", "empty", "nothing", "none"})

# Decoration around a token. NOT the apostrophe: it is a real drum
# character, so "'" must survive token cleanup.
_TOKEN_WRAPPERS = "`\""

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def _chars_from_json(text: str, charset: str) -> list[str] | None:
    """Characters from a JSON reply (models that answer in JSON anyway).

    Accepts ``{"modules": [...]}`` (calib-vlm's tool schema, in case a
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


class TextReader:
    """One photo -> Reading via an OCR prompt (no function calling).

    The photo is encoded here (not upstream) so the image settings —
    ``max_width``, JPEG ``quality``, ``fmt`` — apply to this path only;
    the tool-call path keeps its production-parity encoding.
    """

    def __init__(self, vlm, prompt: str = DEFAULT_OCR_PROMPT,
                 max_width: int = 1024, quality: int = 80,
                 fmt: str = "jpeg", max_tokens: int | None = None):
        self.vlm = vlm
        self.prompt = prompt or DEFAULT_OCR_PROMPT
        self.max_width = max(64, int(max_width))
        self.quality = min(100, max(1, int(quality)))
        self.fmt = "png" if str(fmt).lower() == "png" else "jpeg"
        # Cap generation (None = provider default). An OCR answer needs a
        # handful of tokens; uncapped, a model that loops on a row of one
        # repeated character runs to the server's default limit (~2048
        # tokens, >12 s here) for output the parser never looks at.
        self.max_tokens = int(max_tokens) if max_tokens else None
        self.last_calls = 0
        self.last_raw: str | None = None
        self.last_empty = False

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
                 ocr_max_tokens: int | None = None):
        self.vlm = vlm
        self.mode = mode if mode in ("auto", "tool", "text") else "auto"
        self.tool = VlmReader(vlm, annotate=annotate)
        self.text = TextReader(vlm, ocr_prompt, max_width=image_max_width,
                               quality=image_quality, fmt=image_format,
                               max_tokens=ocr_max_tokens)
        self._effective = "text" if self.mode == "text" else "tool"
        self._reads = 0
        # Row metadata, read by the server after every read() call.
        self.last_mode = self._effective
        self.last_fallback = False
        self.last_empty = False
        self.last_raw: str | None = None

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
        return reading
