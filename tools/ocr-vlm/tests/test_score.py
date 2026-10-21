"""Scoring tests: normalization, positional mismatch, aggregates."""

from __future__ import annotations

from ocr_vlm.score import (mismatches, normalize_char, normalize_text,
                           score_reading, summarize)


# -- normalize_char -----------------------------------------------------------

def test_normalize_char_basics():
    assert normalize_char("A") == "A"
    assert normalize_char("a") == "A"
    assert normalize_char(" l ") == "L"
    assert normalize_char("") == " "
    assert normalize_char(None) == " "
    assert normalize_char("   ") == " "
    assert normalize_char("?") == "?"


def test_normalize_char_aliases():
    assert normalize_char("\u2423") == " "
    assert normalize_char("_") == " "
    assert normalize_char("\u00b7") == " "
    assert normalize_char("space") == " "
    assert normalize_char("blank") == " "


def test_normalize_char_wrappers_but_apostrophe_survives():
    assert normalize_char("'") == "'"          # real drum character
    assert normalize_char("'A'") == "A"        # decoration around one char
    assert normalize_char("`b`") == "B"
    assert normalize_char('"c"') == "C"


# -- normalize_text -----------------------------------------------------------

def test_normalize_text_pads_and_truncates():
    text, adjusted = normalize_text("AB", 4)
    assert text == "AB  " and adjusted is True
    text, adjusted = normalize_text("ABCDEF", 4)
    assert text == "ABCD" and adjusted is True
    text, adjusted = normalize_text("ABCD", 4)
    assert text == "ABCD" and adjusted is False


def test_normalize_text_preserves_spaces():
    # Leading/trailing blanks are real flaps (spaces must never be trimmed).
    text, adjusted = normalize_text("  AB        ", 12)
    assert text == "  AB        " and adjusted is False
    text, adjusted = normalize_text("ab", 4)
    assert text == "AB  "


def test_normalize_text_wrapper_words_and_apostrophes():
    text, adjusted = normalize_text("space", 4)
    assert text == "    " and adjusted is True
    text, adjusted = normalize_text("''''", 4)  # a sweep of apostrophes
    assert text == "''''" and adjusted is False
    text, adjusted = normalize_text('"AB"', 4)
    assert text == "AB  " and adjusted is True
    text, adjusted = normalize_text("_", 2)
    assert text == "  " and adjusted is True


def test_mismatches():
    assert mismatches("ABCD", "ABCD") == 0
    assert mismatches("ABCD", "ABCX") == 1
    assert mismatches("AB  ", "ABXY") == 2


# -- score_reading ------------------------------------------------------------

def test_score_reading_counts_and_flags():
    scored = score_reading("%%%%%%%#%%%%", "%%%%%%%%%%%%",
                           "%%%%%%%#%%%%", 12)
    assert scored["mm_vlm"] == 1
    assert scored["mm_saw"] == 1
    assert scored["adjusted"] is False

    short = score_reading("AB", "AB          ", "AB          ", 12)
    assert short["mm_vlm"] == 0
    assert short["adjusted"] is True      # padded to width
    assert short["read_norm"] == "AB          "

    failed = score_reading(None, "AAAAAAAAAAAA", "AAAAAAAAAAAX", 12)
    assert failed["mm_vlm"] is None
    assert failed["mm_saw"] == 1


# -- summarize ----------------------------------------------------------------

def _row(index: int, photo: str, want: str, saw: str, read: str | None,
         width: int = 12) -> dict:
    scored = score_reading(read, want, saw, width)
    return {"index": index, "photo": photo, "prefix": photo.split("_")[0],
            "want": want, "saw": saw, "read": read, "error": None,
            "flags": [], "modules": [], **scored}


def _error_row(index: int, photo: str) -> dict:
    return {"index": index, "photo": photo, "prefix": photo.split("_")[0],
            "want": "", "saw": "", "read": None, "want_norm": None,
            "saw_norm": None, "read_norm": None, "mm_vlm": None,
            "mm_saw": None, "adjusted": False, "flags": [], "error": "boom",
            "modules": []}


def test_summarize_comparison():
    rows = [
        _row(0, "sw_1_f1.png", "AAAAAAAAAAAA", "AAAAAAAAAAAA", "AAAAAAAAAAAA"),
        _row(1, "sw_2_f2.png", "BBBBBBBBBBBB", "BBBBBBBBBBBB", "BBBXBBBBBBBB"),
        _row(2, "p0_3_f3.png", "CCCCCCCCCCCC", "CCCCCCCCCCCC", "CCCCCCCCCCCC"),
        _error_row(3, "ladder_4_f4.png"),
    ]
    s = summarize(rows, width=12)

    assert s["rows"] == 4
    assert s["errors"] == 1
    assert s["scored_vlm"] == 3 and s["scored_saw"] == 3
    assert s["vlm"]["total"] == 1
    assert s["vlm"]["mean"] == round(1 / 3, 3)
    assert s["vlm"]["exact"] == 2
    assert s["vlm"]["exact_pct"] == round(200 / 3, 1)
    assert s["saw"]["total"] == 0
    assert s["vlm"]["histogram"]["0"] == 2
    assert s["vlm"]["histogram"]["1"] == 1
    # Position 3 is the only VLM mismatch, on 1 of 3 scored rows.
    assert s["vlm"]["per_position"][3] == round(100 / 3, 1)
    assert s["vlm"]["per_position"][0] == 0.0

    h = s["head_to_head"]
    assert (h["n"], h["better"], h["equal"], h["worse"]) == (3, 0, 2, 1)
    assert h["delta_mean"] == round(-1 / 3, 3)

    assert s["confusions"] == [{"want": "B", "got": "X", "count": 1}]
    by_prefix = {p["prefix"]: p for p in s["prefixes"]}
    assert by_prefix["sw"]["rows"] == 2
    assert by_prefix["sw"]["vlm_avg"] == 0.5
    assert by_prefix["p0"]["vlm_avg"] == 0.0
    assert "ladder" not in by_prefix  # error rows are excluded


def test_summarize_head_to_head_sign():
    """delta_mean = saw - vlm, so positive means the VLM did better."""
    rows = [
        _row(0, "a_1.png", "AAAAAAAAAAAA", "AXAAAAAAAAAA", "AAAAAAAAAAAA"),
    ]
    s = summarize(rows, width=12)
    assert s["head_to_head"]["better"] == 1
    assert s["head_to_head"]["delta_mean"] == 1.0
