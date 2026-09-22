"""Text read path tests: OCR parsing and tool->text mode dispatch."""

from __future__ import annotations

import cv2
import numpy as np

from calib_vlm.reader import ModuleReading, ReaderError, Reading

from ocr_vlm.server import DEFAULT_CHARSET
from ocr_vlm.textread import (DEFAULT_OCR_PROMPT, ModeReader, TextReader,
                              cell_crops, extract_single_glyph,
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
    """Captures messages; returns scripted content (one string, or one per
    call when a list is given — the last entry repeats after that)."""

    def __init__(self, content):
        self.contents: list = content if isinstance(content, list) else [content]
        self.calls: list[list[dict]] = []
        self.last_kwargs: dict = {}
        self.last_usage: dict = {}

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        self.last_kwargs = kwargs
        assert not kwargs.get("tools")  # text mode never sends tools
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        return {"content": self.contents[index], "tool_calls": []}


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


def test_mode_reader_passes_image_mode():
    reader = ModeReader(vlm=None, mode="text", image_mode="cells")
    assert reader.text.image_mode == "cells"


# -- cell (per-glyph) segmentation --------------------------------------------

def test_cell_crops_split_into_modules_left_to_right():
    img = np.zeros((4, 120, 3), dtype=np.uint8)
    for i in range(12):
        img[:, i * 10:(i + 1) * 10] = i * 20      # distinct gray per module
    crops = list(cell_crops(img, 12, inset=0.0))
    assert len(crops) == 12
    for i, crop in enumerate(crops):
        assert crop.shape[0] == 4
        assert int(crop[0, 0, 0]) == i * 20       # crop i holds module i


def test_cell_crops_inset_is_clamped():
    img = _img(4, 240)                            # 20 px per module
    crop = next(cell_crops(img, 12, inset=0.9))   # clamps to 20 %
    assert crop.shape[1] == 20 - 2 * 4            # 4 px trimmed each side


def test_extract_single_glyph_shapes():
    assert extract_single_glyph("M", DEFAULT_CHARSET) == ("M", False)
    assert extract_single_glyph(" m ", DEFAULT_CHARSET) == ("M", False)
    assert extract_single_glyph("**M**", DEFAULT_CHARSET) == ("M", False)
    assert extract_single_glyph('%', DEFAULT_CHARSET) == ("%", False)
    assert extract_single_glyph("The character is M",
                                DEFAULT_CHARSET) == ("M", False)
    assert extract_single_glyph("space", DEFAULT_CHARSET) == (" ", False)
    assert extract_single_glyph(None, DEFAULT_CHARSET) == (" ", True)
    assert extract_single_glyph("", DEFAULT_CHARSET) == (" ", True)
    assert extract_single_glyph("```markdown\n\n```",
                                DEFAULT_CHARSET) == (" ", True)


def test_text_reader_cells_mode_maps_every_module():
    script = ["A", "B", None, "D", "E", "F", "G", "H", "I", "J",
              "space", "%"]
    vlm = FakeVlm(script)
    reader = TextReader(vlm, image_mode="cells", max_tokens=64)
    reading = reader.read(_img(260, 1280), total=12, charset=DEFAULT_CHARSET)
    assert reading.text == "AB DEFGHIJ %"        # None/"space" -> blanks
    assert len(vlm.calls) == 12                  # one request per glyph
    assert reader.last_calls == 12               # charged to the call budget
    assert vlm.last_kwargs.get("max_tokens") == 64
    assert reading.realigned is False            # positions are exact
    assert any("1/12 cells" in w for w in reading.warnings)
    assert reading.modules[3].char == "D"
    assert sum(1 for m in reading.modules if m.char == " ") == 2
    # each request carried the configured prompt and one cell-sized image
    prompt_part, image_part_ = vlm.calls[0][0]["content"]
    assert prompt_part["text"] == DEFAULT_OCR_PROMPT
    assert image_part_["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_text_reader_cells_mode_all_empty_reads_blank_row():
    reader = TextReader(FakeVlm([None]), image_mode="cells")
    reading = reader.read(_img(260, 1280), total=12, charset=DEFAULT_CHARSET)
    assert reading.text == " " * 12
    assert reader.last_empty is True


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


# -- detected modes (OpenCV segmentation) -------------------------------------

def _display_img(w: int = 900, h: int = 220, x0: int = 100, x1: int = 700,
                 y0: int = 40, y1: int = 180, total: int = 12,
                 glyphs: str = "", bg: int = 170):
    """Synthetic split-flap frame the detector can localize."""
    img = np.full((h, w, 3), bg, np.uint8)
    img[y0:y1, x0:x1] = 25
    pitch = (x1 - x0) / total
    for i in range(1, total):
        xi = int(round(x0 + i * pitch))
        img[y0:y1, xi - 2:xi + 3] = 8
    cy = (y0 + y1) // 2
    for i, ch in enumerate(glyphs[:total]):
        if ch in (" ", ""):
            continue
        a = int(round(x0 + i * pitch)) + 8
        b = int(round(x0 + (i + 1) * pitch)) - 8
        img[cy - 25:cy + 25, a:b] = 255
    return img


def test_text_reader_montage_is_one_request_with_cv_blank_override():
    img = _display_img(glyphs="AB")
    vlm = FakeVlm("ABXXXXXXXXXX")              # model guesses glyphs on blanks
    reader = TextReader(vlm, image_mode="montage", max_tokens=64)
    reading = reader.read(img, total=12, charset=DEFAULT_CHARSET)
    assert len(vlm.calls) == 1 and reader.last_calls == 1
    assert reading.text == "AB          "       # blanks corrected by OpenCV
    assert any("corrected by OpenCV" in w for w in reading.warnings)
    assert reading.realigned is False
    assert reading.modules[0].source == "text"
    assert reading.modules[5].source == "cv"
    assert reader.last_detected is True
    assert reader.last_display["width"] > 500
    assert reader.last_blanks[0] is False and reader.last_blanks[5] is True
    assert reader.last_composed is not None
    # one montage image is sent, not one image per module
    _, image_part_ = vlm.calls[0][0]["content"]
    assert image_part_["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_text_reader_montage_keeps_parsed_row_when_adjusted():
    img = _display_img(glyphs="AB")
    reader = TextReader(FakeVlm("AB"), image_mode="montage")
    reading = reader.read(img, total=12, charset=DEFAULT_CHARSET)
    assert reading.text == "AB          "
    assert reading.realigned is True            # 2 < 12: padded, flagged
    assert not any("corrected by OpenCV" in w for w in reading.warnings)


def test_text_reader_cells_detect_skips_blank_cells():
    img = _display_img(glyphs="MN")
    vlm = FakeVlm(["M", "N"])
    reader = TextReader(vlm, image_mode="cells-detect", max_tokens=32)
    reading = reader.read(img, total=12, charset=DEFAULT_CHARSET)
    assert reading.text == "MN          "
    assert len(vlm.calls) == 2                  # 10 blank cells never sent
    assert reader.last_calls == 2
    assert any("10/12 blank cells" in w for w in reading.warnings)
    assert reading.realigned is False
    assert reading.modules[1].source == "text"
    assert reading.modules[2].source == "cv"
    assert reader.last_empty is False


def test_text_reader_cells_detect_all_blank_skips_every_call():
    img = _display_img(glyphs="")
    vlm = FakeVlm("M")
    reader = TextReader(vlm, image_mode="cells-detect")
    reading = reader.read(img, total=12, charset=DEFAULT_CHARSET)
    assert reading.text == " " * 12
    assert vlm.calls == [] and reader.last_calls == 0
    assert reader.last_empty is True
    assert any("12/12 blank cells" in w for w in reading.warnings)


def test_text_reader_detected_falls_back_to_strip_without_display():
    vlm = FakeVlm("AB")
    reader = TextReader(vlm, image_mode="montage")
    reading = reader.read(_img(260, 1280), total=12, charset=DEFAULT_CHARSET)
    assert reading.text == "AB          "
    assert len(vlm.calls) == 1
    assert reader.last_no_detect is True
    assert reader.last_detected is False
    assert reader.last_composed is None
    assert any("display not detected" in w for w in reading.warnings)


def test_text_reader_strip_detect_sends_the_styled_display_crop():
    img = _display_img(glyphs="AB")
    vlm = FakeVlm("AB")
    reader = TextReader(vlm, image_mode="strip-detect", preprocess="binary")
    reading = reader.read(img, total=12, charset=DEFAULT_CHARSET)
    assert reading.text == "AB          "
    assert len(vlm.calls) == 1
    data = vlm.calls[0][0]["content"][1]["image_url"]["url"]
    import base64 as _b64
    raw = _b64.b64decode(data.split(",", 1)[1])
    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[1] == 600 and decoded.shape[0] == 140
    assert float(decoded.mean()) > 127          # binary style: black on white
    assert reader.last_detected is True


def test_mode_reader_passes_preprocess_and_detected_metadata():
    reader = ModeReader(vlm=None, mode="text", image_mode="montage",
                        preprocess="binary")
    assert reader.text.preprocess == "binary"
    assert reader.text.image_mode == "montage"
    assert ModeReader(vlm=None, mode="text",
                      preprocess="bogus").text.preprocess == "none"


def test_mode_reader_auto_fallback_carries_detected_metadata():
    tool = FakeTool(error=ReaderError("no tool call"))
    text = FakeText()
    text.last_no_detect = True
    text.last_detected = False
    text.last_display = {"width": 10}
    text.last_blanks = [True] * 12
    text.last_composed = None
    reader = _mode_reader("auto", tool, text)
    reader.read(_img(), 12, charset=DEFAULT_CHARSET)
    assert reader.last_mode == "text"
    assert reader.last_no_detect is True
    assert reader.last_detected is False
    assert reader.last_display == {"width": 10}
    assert reader.last_blanks == [True] * 12
