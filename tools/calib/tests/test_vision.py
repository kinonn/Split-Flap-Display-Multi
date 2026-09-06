"""Synthetic fixtures + tests for calib/vision.py (no camera needed)."""

from tests.fixtures import make_crop, make_frame, make_glyph

from calib import vision


def test_clean_scores_ok():
    assert vision.score_crop(make_crop("clean"))["verdict"] == "ok"


def test_half_flap_detected():
    score = vision.score_crop(make_crop("half"))
    assert score["verdict"] == "half"
    assert 0.35 <= score["seam_pos"] <= 0.65


def test_double_flap_detected():
    assert vision.score_crop(make_crop("double"))["verdict"] == "double"


def test_split_crops_count_and_width():
    frame = make_frame(4)
    crops = vision.split_crops(frame, 4)
    assert len(crops) == 4
    for c in crops:
        assert 50 <= c.shape[1] <= 78  # near nominal 68 incl. gap share


def test_stuck_detection():
    a = make_crop("clean")
    assert vision.stuck(a, a.copy()) is True
    assert vision.stuck(a, make_crop("half")) is False


def test_zncc_identical_near_one():
    a = make_glyph("E")
    assert vision.zncc(a, a.copy()) > 0.99


def test_zncc_distinct_glyphs_low():
    assert vision.zncc(make_glyph("E"), make_glyph("I")) < 0.85


def test_consensus_flags_wrong_glyph():
    crops = [make_glyph("E"), make_glyph("E"), make_glyph("E"), make_glyph("I")]
    assert vision.consensus_outliers(crops) == [3]


def test_consensus_all_agree():
    assert vision.consensus_outliers([make_glyph("E")] * 4) == []


def test_consensus_abstains_below_three():
    assert vision.consensus_outliers([make_glyph("E"), make_glyph("I")]) == []


def test_build_bank_and_identify():
    bank = vision.build_template_bank({"E": [make_glyph("E")] * 3,
                                       "I": [make_glyph("I")] * 3})
    assert set(bank) == {"E", "I"}
    assert vision.identify(make_glyph("E"), bank)[0][0] == "E"
    assert vision.identify(make_glyph("I"), bank)[0][0] == "I"


def test_build_bank_median_survives_bad_sample():
    bank = vision.build_template_bank(
        {"E": [make_glyph("E"), make_glyph("E"), make_glyph("I")]})
    assert vision.identify(make_glyph("E"), bank)[0][0] == "E"


def test_save_load_bank_roundtrip(tmp_path):
    bank = vision.build_template_bank({"E": [make_glyph("E")] * 2})
    vision.save_template_bank(bank, {"source": "test"}, str(tmp_path))
    loaded, manifest = vision.load_template_bank(str(tmp_path))
    assert manifest["source"] == "test"
    assert vision.identify(make_glyph("E"), loaded)[0][0] == "E"
