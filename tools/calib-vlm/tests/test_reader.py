"""Reader tests: schema parsing, space reconciliation, retries."""

import numpy as np
import pytest

from calib_vlm.reader import ReaderError, VlmReader, annotate_modules

CHARSET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def entry(ch, cond="clean", conf=0.9, module=0):
    return {"module": module, "char": ch, "condition": cond,
            "confidence": conf}


def tool_reply(entries):
    return {"content": None,
            "tool_calls": [{"id": "c1", "name": "report_reading",
                            "arguments": {"modules": entries}}]}


class FakeVlm:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=None, tool_choice=None):
        self.calls.append(messages)
        if not self.replies:
            raise AssertionError("no scripted replies left")
        return self.replies.pop(0)


def make_reader(replies):
    vlm = FakeVlm(replies)
    return VlmReader(vlm, annotate=False), vlm


def test_exact_read():
    entries = [entry("H", module=i) for i in range(4)]
    reader, _ = make_reader([tool_reply(entries)])
    reading = reader.read(b"j", total=4, expected="HHHH", charset=CHARSET,
                          drum=CHARSET)
    assert reading.text == "HHHH"
    assert not reading.realigned
    assert all(m.condition == "clean" for m in reading.modules)
    assert reading.raw_count == 4


def test_compacted_blanks_realigned_against_expected():
    # Display shows "  ABC  " but the model returned only A, B, C: the
    # visible run must land at offset 2 and blanks be inferred.
    entries = [entry("A"), entry("B"), entry("C")]
    reader, _ = make_reader([tool_reply(entries)])
    reading = reader.read(b"j", total=7, expected="  ABC  ", charset=CHARSET,
                          drum=CHARSET)
    assert reading.text == "  ABC  "
    assert reading.realigned
    assert reading.modules[0].source == "inferred"
    assert reading.modules[0].condition == "blank"
    assert reading.modules[5].source == "inferred"
    assert reading.modules[2].source == "vlm"


def test_wrong_length_retried_then_exact():
    first = [entry(ch, module=i) for i, ch in enumerate("  ABC  XYZ")]
    second = [entry(ch, module=i) for i, ch in enumerate("  ABC  ")]
    reader, vlm = make_reader([tool_reply(first), tool_reply(second)])
    reading = reader.read(b"j", total=7, expected="  ABC  ", charset=CHARSET,
                          drum=CHARSET)
    assert len(vlm.calls) == 2
    assert reading.text == "  ABC  "
    assert not reading.realigned


def test_no_tool_call_raises():
    reader, _ = make_reader([{"content": "hello", "tool_calls": []},
                             {"content": "still no", "tool_calls": []}])
    with pytest.raises(ReaderError):
        reader.read(b"j", total=4, expected="    ", charset=CHARSET,
                    drum=CHARSET)


def test_condition_and_char_normalization():
    entries = [entry("h", "OK", 80), entry("space", "empty", 150),
               entry("~", "weird", 0.3), entry("H", "half-flap", 0.5)]
    reader, _ = make_reader([tool_reply(entries)])
    reading = reader.read(b"j", total=4, expected="HO?H", charset=CHARSET,
                          drum=CHARSET)
    assert reading.modules[0].char == "H"       # lowercased made uppercase
    assert reading.modules[0].condition == "clean"
    assert reading.modules[0].confidence == pytest.approx(0.8)
    assert reading.modules[1].char == " "
    assert reading.modules[1].condition == "blank"
    assert reading.modules[1].confidence == pytest.approx(1.0)
    assert reading.modules[2].char == "?"       # not in the charset
    assert reading.modules[2].condition == "unreadable"
    assert reading.modules[3].condition == "half"


def test_error_reading_is_all_unreadable():
    reader, _ = make_reader([])
    reading = reader.error_reading(3, "ABC", "boom")
    assert reading.text == "???"
    assert all(m.source == "error" for m in reading.modules)
    assert reading.warnings == ["boom"]


def test_annotate_modules_draws_grid():
    img = np.zeros((40, 120, 3), dtype=np.uint8)
    out = annotate_modules(img, 4)
    assert out.shape[0] > img.shape[0]   # top band for index labels
    assert out.shape[1] == img.shape[1]
    assert out.max() > 0                 # lines/labels were drawn
