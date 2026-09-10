"""VlmCalibrator tests: delta math, convergence, escalation, fleet."""

import math

from calib_vlm.calibrate import VlmCalibrator

from tests.fixtures import FakeCamera, FakeDisplay, SimReader


def run_calib(display, tmp_path, mode="full", exhaustive=False):
    calib = VlmCalibrator(display, FakeCamera(), SimReader(display),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, exhaustive=exhaustive, mode=mode)
    return calib, calib.run()


def test_coarse_identity_converges(tmp_path):
    d = FakeDisplay(total=4)
    d.seed_module_error(1, -d.spc)  # module 1 shows the previous char
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert report["summary"]["ok"]
    assert d.mod_off[1] == 0
    assert d.persists
    # A whole-drum fault is tuned on the coarse module cell.
    m1 = [x for x in report["deltas"] if x["globalModule"] == 1]
    assert any(x["charIndex"] == -1 for x in m1)
    # Every preview must respect the firmware's per-call delta limit.
    assert d.previews
    assert all(abs(delta) <= 32 for _, _, delta in d.previews)


def test_per_char_identity_converges(tmp_path):
    d = FakeDisplay(total=4)
    ci = d.drum.index("O")
    d.seed_char_error(2, ci, -d.spc)  # only 'O' is one char behind on m2
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.char_off[2].get(ci, 0) == 0
    # A single-glyph fault must stay on a char cell; a coarse module shift
    # would break every other character on that drum.
    m2 = [x for x in report["deltas"] if x["globalModule"] == 2]
    assert m2 and all(x["charIndex"] >= 0 for x in m2)
    assert all(abs(delta) <= 32 for _, _, delta in d.previews)


def test_per_char_identity_overflow_escalates_cleanly(tmp_path):
    # Reading one char AHEAD sends the forward-only fix 36 characters
    # around the drum (1980 steps), which cannot live in a char cell the
    # firmware clamps to ±32: escalate instead of writing a clamped,
    # wrong offset that would leave the display worse than before.
    d = FakeDisplay(total=4)
    ci = d.drum.index("O")
    d.seed_char_error(2, ci, d.spc)  # 'O' shows the NEXT char on m2
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "needs-human"
    assert d.char_off[2].get(ci, 0) == d.spc   # cell untouched, not clamped
    assert not d.previews and not d.persists
    notes = [e["note"] for e in report["identity"]["persistent"]]
    assert any("does not fit a char cell" in n for n in notes)


def test_remote_per_char_overflow_escalates_cleanly(tmp_path):
    d = FakeDisplay(total=6, groups=2)
    ci = d.drum.index("H")
    d.seed_char_error(5, ci, d.spc)  # group 2, local 2, char H
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "needs-human"
    assert d.remote_char[0][2][ci] == d.spc    # not corrupted to ±32
    notes = [e["note"] for e in report["identity"]["persistent"]]
    assert any("does not fit a char cell" in n for n in notes)


def test_alignment_half_flap_converges_exhaustive(tmp_path):
    d = FakeDisplay(total=4)
    ci = d.drum.index("A")
    d.seed_char_error(3, ci, 1)  # same glyph, one motor step out of phase
    calib, report = run_calib(d, tmp_path, exhaustive=True)
    assert report["result"] == "converged"
    assert d.char_off[3].get(ci, 0) == 0


def test_remote_group_converges(tmp_path):
    d = FakeDisplay(total=6, groups=2)
    d.seed_module_error(4, -d.spc)  # group 2, local module 1
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.remote_mod[0][1] == 0


def test_remote_group_char_converges(tmp_path):
    d = FakeDisplay(total=6, groups=2)
    ci = d.drum.index("H")
    d.seed_char_error(5, ci, -d.spc)  # group 2, local 2, char H
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.remote_char[0][2][ci] == 0

def test_stuck_module_escalates(tmp_path):
    d = FakeDisplay(total=4)
    d.seed_module_error(2, -d.spc)
    reader = SimReader(d)
    reader.frozen[2] = "X"  # camera sees this module never move
    calib = VlmCalibrator(d, FakeCamera(), reader, photo_dir=str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full")
    report = calib.run()
    assert report["result"] == "needs-human"
    assert any(e["module"] == 2
               for e in report["identity"]["persistent"])


def test_dry_run_proposes_without_touching(tmp_path):
    d = FakeDisplay(total=4)
    d.seed_module_error(1, -d.spc)
    _, report = run_calib(d, tmp_path, mode="dry-run")
    assert report["result"] == "needs-human"
    assert d.mod_off[1] == -d.spc          # nothing applied
    assert not d.persists
    assert not d.previews
    assert any(e.get("proposal") for e in report["deltas"])


def test_preview_mode_never_persists(tmp_path):
    d = FakeDisplay(total=4)
    d.seed_module_error(1, -d.spc)
    _, report = run_calib(d, tmp_path, mode="preview")
    assert not d.persists
    assert d.previews


def test_p1_coarse_glyphs_cover_quarter_of_drum(tmp_path):
    for charset in (37, 48):
        d = FakeDisplay(total=4, charset=charset)
        calib = VlmCalibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                              dwell_ms=0, timeout_s=5, min_confidence=0.5,
                              mode="full")
        calib.drum = d.drum
        calib.total = d.total
        glyphs = calib._coarse_glyphs()
        assert len(glyphs) >= math.ceil(len(d.drum) * 0.25)
        assert len(set(glyphs)) == len(glyphs)   # no duplicates
        assert " " not in glyphs                 # blanks prove nothing
        assert all(g in d.drum for g in glyphs)


def test_summary_records_offset_delta(tmp_path):
    d = FakeDisplay(total=4)
    d.seed_module_error(1, -d.spc)
    _, report = run_calib(d, tmp_path)
    m1 = [m for m in report["summary"]["modules"] if m["module"] == 1][0]
    assert m1["offset_delta"] == d.spc
    assert not m1["persistent_wrong_glyph"]


def test_report_written_to_disk(tmp_path):
    d = FakeDisplay(total=4)
    run_calib(d, tmp_path)
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "snapshot.json").is_file()
