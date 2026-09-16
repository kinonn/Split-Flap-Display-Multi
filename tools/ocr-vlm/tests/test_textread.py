"""Text read path tests: OCR parsing and tool->text mode dispatch."""

from __future__ import annotations

import cv2
import numpy as np

from calib_vlm.reader import ModuleReading, ReaderError, Reading

from ocr_vlm.server import DEFAULT_CHARSET
from ocr_vlm.textread import (DEFAULT_OCR_PROMPT, ModeReader, TextReader,
                              parse_ocr_text)


def _img(h: int = 6, w: int = 48):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _decoded_hw(jpeg: bytes) -> tuple[int, int]:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return img.shape[0], img.shape[1]


def _reading(chars: str, source: str = "vlm") -> Reading:
    modules = [ModuleReading(i, ch, "blank" if ch == " " else "clean",
                             1.0, source, " ")
               for i, ch in enumerate(chars)]
    return Reading(modules)


# -- parse_ocr_text -----------------------------------------------------------

def test_parse_space_separated_characters():
    # The observed PaddleOCR-VL output for a 12-module frame.
    chars, raw_len, warnings = parse_ocr_text(
        "% % % % % % % # % % % %", 12, DEFAULT_CHARSET)
    assert "".join(chars) == "%%%%%%%#%%%%"
    assert raw_len == 12 and warnings == []


def test_parse_contiguous_and_merged_chunks():
    chars, raw_len, _ = parse_ocr_text("ABCDEFGHIJKL", 12, DEFAULT_CHARSET)
    assert "".join(chars) == "ABCDEFGHIJKL" and raw_len == 12

    chars, raw_len, warnings = parse_ocr_text(
        "MMMJN% % BBCN", 12, DEFAULT_CHARSET)
    assert "".join(chars) == "MMMJN%%BBCN "   # 11 chars, padded to 12
    assert raw_len == 11
    assert any("padded" in w for w in warnings)


def test_parse_empty_reply_reads_as_blanks():
    chars, raw_len, warnings = parse_ocr_text(None, 12, DEFAULT_CHARSET)
    assert "".join(chars) == " " * 12
    assert raw_len == 0
    assert any("empty reply" in w for w in warnings)


def test_parse_blank_words_and_aliases():
    chars, _, _ = parse_ocr_text("A space B", 12, DEFAULT_CHARSET)
    assert "".join(chars) == "A B         "
    chars, _, _ = parse_ocr_text("\u2423 A", 4, DEFAULT_CHARSET)
    assert "".join(chars) == " A  "


def test_parse_apostrophes_survive_but_wrappers_are_removed():
    chars, _, _ = parse_ocr_text("`%` \"@\"", 12, DEFAULT_CHARSET)
    assert "".join(chars) == "%@          "
    # A run of apostrophes is real content (the drum has an apostrophe),
    # so token cleanup must not eat it.
    chars, _, _ = parse_ocr_text("''''", 4, DEFAULT_CHARSET)
    assert "".join(chars) == "''''"


def test_parse_lowercase_is_uppercased_unknown_becomes_question():
    chars, _, _ = parse_ocr_text("ab\u00e9", 4, DEFAULT_CHARSET)
    assert "".join(chars) == "AB? "


def test_parse_truncates_overlong_reply():
    chars, raw_len, warnings = parse_ocr_text(
        "ABCDEFGHIJKLMNOP", 12, DEFAULT_CHARSET)
    assert "".join(chars) == "ABCDEFGHIJKL"
    assert raw_len == 16
    assert any("truncated" in w for w in warnings)


def test_parse_json_replies():
    chars, _, _ = parse_ocr_text('["%", "%", "@"]', 6, DEFAULT_CHARSET)
    assert "".join(chars) == "%%@   "
    chars, _, _ = parse_ocr_text(
        '{"modules": [{"char": "A"}, {"char": "B"}]}', 4, DEFAULT_CHARSET)
    assert "".join(chars) == "AB  "
    chars, _, _ = parse_ocr_text(
        "```json\n{\"text\": \"AB\"}\n```", 4, DEFAULT_CHARSET)
    assert "".join(chars) == "AB  "


# -- TextReader ---------------------------------------------------------------

class FakeVlm:
    """Captures messages; returns one scripted content string."""

    def __init__(self, content):
        self.content = content
        self.calls: list[list[dict]] = []
        self.last_kwargs: dict = {}
        self.last_usage: dict = {}

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        self.last_kwargs = kwargs
        assert not kwargs.get("tools")  # text mode never sends tools
        return {"content": self.content, "tool_calls": []}


def test_text_reader_builds_reading_without_tools():
    vlm = FakeVlm("% % % % % % % # % % % %")
    reader = TextReader(vlm)
    reading = reader.read(_img(), total=12, expected="", charset=DEFAULT_CHARSET)
    assert reading.text == "%%%%%%%#%%%%"
    assert len(vlm.calls) == 1
    assert vlm.calls[0][0]["role"] == "user"
    prompt_part, image_part_ = vlm.calls[0][0]["content"]
    assert prompt_part["text"] == DEFAULT_OCR_PROMPT
    assert image_part_["type"] == "image_url"
    assert image_part_["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert all(m.source == "text" for m in reading.modules)
    assert reader.last_raw == "% % % % % % % # % % % %"
    assert reader.last_empty is False


def test_text_reader_empty_reply_marks_empty():
    reader = TextReader(FakeVlm(None))
    reading = reader.read(_img(), total=12, charset=DEFAULT_CHARSET)
    assert reading.text == " " * 12
    assert reader.last_empty is True
    assert any("empty reply" in w for w in reading.warnings)


def test_text_reader_custom_prompt():
    vlm = FakeVlm("AB")
    TextReader(vlm, "Text Recognition:").read(_img(), total=2,
                                              charset=DEFAULT_CHARSET)
    assert vlm.calls[0][0]["content"][0]["text"] == "Text Recognition:"


def test_text_reader_forwards_max_tokens_cap():
    """A repetition-looping model must be stopped by the request cap:
    the parser only reads the first characters, so capping generation
    changes scores not at all (greedy prefix) but saves seconds a row."""
    vlm = FakeVlm("AB")
    reader = TextReader(vlm, max_tokens=64)
    reader.read(_img(), total=2, charset=DEFAULT_CHARSET)
    assert vlm.last_kwargs.get("max_tokens") == 64


def test_text_reader_default_is_uncapped():
    vlm = FakeVlm("AB")
    reader = TextReader(vlm)
    reader.read(_img(), total=2, charset=DEFAULT_CHARSET)
    assert reader.max_tokens is None
    assert vlm.last_kwargs.get("max_tokens") is None


def test_mode_reader_passes_ocr_max_tokens():
    reader = ModeReader(vlm=None, mode="text", ocr_max_tokens=32)
    assert reader.text.max_tokens == 32


def test_text_reader_converts_vlm_errors():
    class Boom(FakeVlm):
        def chat(self, messages, **kwargs):
            from calib_vlm.vlm import VLMError
            raise VLMError("HTTP 500")

    reader = TextReader(Boom(None))
    try:
        reader.read(_img(), total=12, charset=DEFAULT_CHARSET)
    except ReaderError as exc:
        assert "HTTP 500" in str(exc)
    else:
        raise AssertionError("expected ReaderError")


# -- ModeReader ---------------------------------------------------------------

class FakeTool:
    annotate = True   # real VlmReader exposes this; gates image annotation

    def __init__(self, error: ReaderError | None = None, text: str = "AB"):
        self.error = error
        self.text = text
        self.calls = 0
        self.last_jpeg: bytes | None = None

    def read(self, jpeg, total, expected="", charset="", drum=""):
        self.calls += 1
        self.last_jpeg = jpeg
        if self.error:
            raise self.error
        return _reading(self.text.ljust(total)[:total])


class FakeText:
    def __init__(self, text: str = "ABCDEFGHIJKL"):
        self.text = text
        self.calls = 0
        self.last_image = None
        self.last_raw = "raw:" + text
        self.last_empty = False

    def read(self, image, total, expected="", charset="", drum=""):
        self.calls += 1
        self.last_image = image
        return _reading(self.text.ljust(total)[:total], source="text")


def _mode_reader(mode: str, tool: FakeTool, text: FakeText) -> ModeReader:
    reader = ModeReader(vlm=None, mode=mode)
    reader.tool = tool
    reader.text = text
    return reader


def test_mode_tool_uses_tool_reader():
    tool, text = FakeTool(text="AB"), FakeText()
    reader = _mode_reader("tool", tool, text)
    reading = reader.read(_img(), 12, charset=DEFAULT_CHARSET)
    assert reading.text == "AB          "
    assert reader.last_mode == "tool" and text.calls == 0


def test_mode_tool_propagates_reader_error():
    tool = FakeTool(error=ReaderError("no tool call"))
    reader = _mode_reader("tool", tool, FakeText())
    try:
        reader.read(_img(), 12, charset=DEFAULT_CHARSET)
    except ReaderError:
        pass
    else:
        raise AssertionError("expected ReaderError")


def test_mode_text_never_calls_tool():
    tool, text = FakeTool(), FakeText()
    reader = _mode_reader("text", tool, text)
    reading = reader.read(_img(), 12, charset=DEFAULT_CHARSET)
    assert reading.text == "ABCDEFGHIJKL"
    assert reader.last_mode == "text" and tool.calls == 0
    assert reader.last_raw == "raw:ABCDEFGHIJKL"
    assert reader.last_fallback is False


def test_auto_probes_then_sticks_to_text():
    tool = FakeTool(error=ReaderError("model did not call report_reading"))
    text = FakeText()
    reader = _mode_reader("auto", tool, text)

    reading = reader.read(_img(), 12, charset=DEFAULT_CHARSET)
    assert reading.text == "ABCDEFGHIJKL"
    assert reader.last_mode == "text" and reader.last_fallback is True
    assert any("text mode" in w for w in reading.warnings)
    assert tool.calls == 1 and text.calls == 1

    # The next read goes straight to text: no second probe.
    reading = reader.read(_img(), 12, charset=DEFAULT_CHARSET)
    assert reader.last_fallback is False and reader.last_mode == "text"
    assert tool.calls == 1 and text.calls == 2


def test_auto_keeps_tool_after_success_then_errors():
    tool = FakeTool(text="AB")
    text = FakeText()
    reader = _mode_reader("auto", tool, text)
    reader.read(_img(), 12, charset=DEFAULT_CHARSET)      # probe succeeds
    assert reader.last_mode == "tool"

    tool.error = ReaderError("bad frame")
    try:
        reader.read(_img(), 12, charset=DEFAULT_CHARSET)
    except ReaderError:
        pass
    else:
        raise AssertionError("expected ReaderError once tool mode is settled")
    assert text.calls == 0


# -- image preparation per path -----------------------------------------------

def test_text_path_gets_the_untouched_photo():
    """A probed PaddleOCR-VL server OCRs the annotated index ticks
    ('0 1 2 ... 11') instead of the display, so text mode must never
    send the module-grid annotation."""
    tool, text = FakeTool(), FakeText()
    _mode_reader("text", tool, text).read(_img(), 12, charset=DEFAULT_CHARSET)
    assert text.last_image.shape[:2] == (6, 48)


def test_tool_path_gets_the_annotated_photo():
    tool, text = FakeTool(text="AB"), FakeText()
    _mode_reader("tool", tool, text).read(_img(), 12, charset=DEFAULT_CHARSET)
    annotated = _decoded_hw(tool.last_jpeg)
    assert annotated[0] > 6 and annotated[1] == 48   # index-tick border added


def test_auto_fallback_reads_the_untouched_photo():
    tool = FakeTool(error=ReaderError("no tool call"))
    text = FakeText()
    _mode_reader("auto", tool, text).read(_img(), 12, charset=DEFAULT_CHARSET)
    assert _decoded_hw(tool.last_jpeg)[0] > 6          # probe was annotated
    assert text.last_image.shape[:2] == (6, 48)        # fallback is not


# -- image settings (text path) -----------------------------------------------

def test_text_encode_downscales_to_default_1024():
    reader = TextReader(FakeVlm(""))
    data, media = reader.encode(_img(300, 2000))
    assert media == "image/jpeg"
    assert _decoded_hw(data)[1] == 1024


def test_text_encode_honours_configured_width():
    reader = TextReader(FakeVlm(""), max_width=1280, quality=95)
    data, _ = reader.encode(_img(260, 1280))
    assert _decoded_hw(data) == (260, 1280)     # native kept, no downscale


def test_text_encode_png_is_lossless_and_typed():
    reader = TextReader(FakeVlm(""), fmt="png", max_width=1280)
    data, media = reader.encode(_img(260, 1280))
    assert media == "image/png"
    assert _decoded_hw(data) == (260, 1280)


def test_text_read_sends_configured_format():
    vlm = FakeVlm("AB")
    reader = TextReader(vlm, fmt="png", max_width=2048)
    reader.read(_img(), total=2, charset=DEFAULT_CHARSET)
    _, image_part_ = vlm.calls[0][0]["content"]
    assert image_part_["image_url"]["url"].startswith("data:image/png;base64,")
