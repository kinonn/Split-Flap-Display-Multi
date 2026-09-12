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
    # Char-cell previews respect the firmware's per-call ±32 clamp; module
    # cells are unbounded and applied in one shot (see the single-preview
    # regression test).
    assert d.previews
    assert all(abs(delta) <= 32 for _, ci, delta in d.previews if ci >= 0)


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
    assert all(abs(delta) <= 32 for _, ci, delta in d.previews if ci >= 0)


def test_per_char_identity_overflow_escalates_cleanly(tmp_path):
    # A fault needing MORE than half a revolution (e.g. 30 chars on a
    # 48-char drum) cannot live in a char cell the firmware clamps to
    # ±32: escalate instead of writing a clamped, wrong offset that
    # would leave the display worse than before. Driven directly (a
    # full run would route a many-glyph fault to the module cell).
    d = FakeDisplay(total=4)
    ci = d.drum.index("O")
    far = d.drum[(ci + 30) % len(d.drum)]
    reader = SimReader(d)
    reader.frozen[2] = far
    calib = VlmCalibrator(d, FakeCamera(), reader, photo_dir=str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full")
    calib.total, calib.charset, calib.drum = d.total, d.charset, d.drum
    calib.group_widths = [d.total]
    calib.steps_per_char = d.spc
    out = calib._tune_identity(2, ci, "O", "O" * d.total)
    assert out["fixed"] is False
    assert d.char_off[2].get(ci, 0) == 0   # cell untouched, not clamped
    notes = [e["note"] for e in calib.identity_persistent]
    assert any("does not fit a char cell" in n for n in notes)


def test_remote_per_char_overflow_escalates_cleanly(tmp_path):
    d = FakeDisplay(total=6, groups=2)
    ci = d.drum.index("H")
    far = d.drum[(ci + 30) % len(d.drum)]
    reader = SimReader(d)
    reader.frozen[5] = far  # group 2, local 2: 30 chars ahead
    calib = VlmCalibrator(d, FakeCamera(), reader, photo_dir=str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full")
    calib.total, calib.charset, calib.drum = d.total, d.charset, d.drum
    calib.group_widths = d.widths()
    calib.steps_per_char = d.spc
    calib._load_remote_offsets(calib.display.snapshot().get("settings", {}))
    out = calib._tune_identity(5, ci, "H", "H" * d.total)
    assert out["fixed"] is False
    assert d.remote_char[0][2][ci] == 0    # not corrupted to ±32
    notes = [e["note"] for e in calib.identity_persistent]
    assert any("does not fit a char cell" in n for n in notes)


def test_alignment_half_flap_converges_exhaustive(tmp_path):
    d = FakeDisplay(total=4)
    ci = d.drum.index("A")
    d.seed_char_error(3, ci, 1)  # same glyph, one motor step out of phase
    calib, report = run_calib(d, tmp_path, exhaustive=True)
    assert report["result"] == "converged"
    assert d.char_off[3].get(ci, 0) == 0


def test_module_trim_fixes_boundary_flaps(tmp_path):
    # Manual process, layer 0: a module wrong on only a few glyphs is a
    # boundary/phase problem, not several broken characters. A sub-pitch
    # module trim pulls those flaps back without touching char cells and
    # without moving the centred characters.
    d = FakeDisplay(total=4, charset=48)
    d.flap_window = int(round(d.spc * 0.4))   # readable off-centre window
    for glyph in ("D", "H"):
        d.seed_flap_error(1, d.drum.index(glyph), int(round(d.spc * 0.6)))
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.mod_off[1] != 0                  # fixed on the module cell
    assert all(not row for row in d.char_off)  # no per-char cell touched
    m1 = [x for x in report["deltas"] if x["globalModule"] == 1]
    assert any(x["charIndex"] == -1 for x in m1)
    assert all(x["charIndex"] == -1 for x in m1)


def test_sweep_is_reverse_drum_order(tmp_path):
    # The sweep walks the drum backwards so every step is ~a full revolution
    # (per-frame magnet re-home). Order and coverage are part of the design.
    d = FakeDisplay(total=4, charset=48)
    seen = []
    orig = d.show_and_settle

    def counting(frame, *a, **k):
        seen.append(frame)
        return orig(frame, *a, **k)

    d.show_and_settle = counting
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full")
    calib.total, calib.charset, calib.drum = d.total, d.charset, d.drum
    readings, chars = calib._sweep()
    assert chars == list(reversed(d.drum))
    assert seen == [ch * d.total for ch in reversed(d.drum)]


def test_single_boundary_flap_fixed_at_module_level(tmp_path):
    # A lone flap just past its boundary is still a module-phase problem:
    # the sub-pitch trim fixes it without a char cell.
    d = FakeDisplay(total=4, charset=48)
    d.flap_window = int(round(d.spc * 0.4))
    d.seed_flap_error(2, d.drum.index("T"), int(round(d.spc * 0.6)))
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.mod_off[2] != 0
    assert all(not row for row in d.char_off)


def test_module_trim_trims_modules_in_parallel_batches(tmp_path):
    # Independent modules are adjusted in the same rounds: one batch per
    # round carries every suspect module's candidate, so the frame count
    # does not grow with the number of modules.
    d = FakeDisplay(total=4, charset=48)
    d.flap_window = int(round(d.spc * 0.4))
    d.seed_flap_error(1, d.drum.index("H"), int(round(d.spc * 0.6)))
    d.seed_flap_error(3, d.drum.index("O"), int(round(d.spc * 0.6)))
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    multi = [b for b in d.batches if len(b) >= 2]
    assert multi, "expected a batch carrying both modules' candidates"
    # One candidate batch per trim round (plus reverts), far below the
    # serial one-batch-per-module-per-round cost.
    assert len(d.batches) <= 24


def test_module_trim_fixes_remote_boundary_flaps(tmp_path):
    # Remote groups are trimmed the same way: the master forwards the
    # volatile nudge over ESP-NOW (no NVS write) and the group is only
    # persisted once the trim actually improved it.
    d = FakeDisplay(total=6, groups=2, charset=48)
    d.flap_window = int(round(d.spc * 0.4))
    d.seed_flap_error(4, d.drum.index("H"), int(round(d.spc * 0.6)))  # g2 m0
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert any(n.get("scope") == 2 for batch in d.batches for n in batch)
    assert d.remote_mod[0][1] != 0
    assert all(not row for row in d.char_off)


def test_module_trim_skips_mixed_direction_faults(tmp_path):
    # One flap ahead and one behind cannot be fixed by a single whole-drum
    # shift: the trim must not touch that module.
    d = FakeDisplay(total=4, charset=48)
    d.flap_window = int(round(d.spc * 0.4))
    d.seed_flap_error(1, d.drum.index("H"), int(round(d.spc * 0.6)))
    d.seed_flap_error(1, d.drum.index("D"), -int(round(d.spc * 0.6)))
    calib, report = run_calib(d, tmp_path)
    assert all(n["module"] != 1 for batch in d.batches for n in batch)


def test_module_trim_dry_run_touches_nothing(tmp_path):
    # Regression: the trim bypassed mode gating, so dry-run applied
    # volatile previews (and even persisted remote trims) despite the
    # read-only contract. Dry-run must leave every offset untouched.
    d = FakeDisplay(total=4, charset=48)
    d.flap_window = int(round(d.spc * 0.4))
    d.seed_flap_error(1, d.drum.index("H"), int(round(d.spc * 0.6)))
    calib, report = run_calib(d, tmp_path, mode="dry-run")
    assert not d.previews
    assert not d.persists
    assert not d.batches
    assert d.mod_off[1] == 0
    assert d.res_mod[1] == 0


def test_module_phase_fault_fixed_at_module_level(tmp_path):
    # Manual process, layer 1: a whole-drum phase error (every glyph
    # reads right but half-flap) gets ONE module-cell alignment search,
    # not a per-char search per glyph.
    d = FakeDisplay(total=4)
    d.seed_module_error(1, 2)  # every glyph 2 motor steps out of phase
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.mod_off[1] == 0
    m1 = [x for x in report["deltas"] if x["globalModule"] == 1]
    assert m1 and all(x["charIndex"] == -1 for x in m1)


def test_module_fix_clears_char_suspects_without_char_tunes(tmp_path):
    # Manual process, layer 2: after a module-cell fix, P2 suspects on
    # that module re-verify clean — no char cell is ever touched.
    d = FakeDisplay(total=4)
    d.seed_module_error(1, -d.spc)  # whole drum off by one char
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    m1 = [x for x in report["deltas"] if x["globalModule"] == 1]
    assert m1 and all(x["charIndex"] == -1 for x in m1)
    assert 1 not in calib.module_fixed  # re-check cleared it


def test_remote_group_converges(tmp_path):
    d = FakeDisplay(total=6, groups=2)
    d.seed_module_error(4, -d.spc)  # group 2, local module 1
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.remote_mod[0][1] == 0


def test_ahead_by_one_char_converges_small_negative(tmp_path):
    # The aborted-run regression: commanded E, showed F (one char AHEAD).
    # Signed-minimal fix is -steps_per_char (2 chunks), not a 47-char
    # near-full revolution forward (was 64 chunks / ~5 min of grinding).
    d = FakeDisplay(total=4)
    ci = d.drum.index("E")
    d.seed_char_error(1, ci, d.spc)  # 'E' shows the NEXT char on m1
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.char_off[1].get(ci, 0) == 0
    char_previews = [(c, d_) for _, c, d_ in d.previews if c >= 0]
    assert all(abs(delta) <= 32 for _, delta in char_previews)
    assert len(char_previews) <= 2  # -spc fits in two <=32 chunks, not 64


def test_module_cell_fault_applied_in_one_preview(tmp_path):
    # Regression: a whole-drum (module-cell) fault must be corrected with the
    # sign the firmware actually uses and in a SINGLE preview. The old code
    # used the opposite sign and chunked at 32, so a 12-char error ground
    # through ~30 re-homes walking the drum AWAY from the target (this is the
    # aborted run-001: Q -> '1' -> 'C' instead of Q -> E).
    d = FakeDisplay(total=4)
    d.seed_module_error(1, -d.spc * 12)  # every glyph 12 chars off
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.mod_off[1] == 0
    mod_previews = [(ci, delta) for _, ci, delta in d.previews if ci < 0]
    assert len(mod_previews) == 1          # one re-home, not 30+
    assert mod_previews[0][1] == -12 * d.spc


def test_run_reverts_volatile_preview_residue(tmp_path):
    # Regression for the run-001 baseline corruption: a previous aborted or
    # preview run leaves RAM-only preview residue that never reverts (the
    # settings rollback re-POSTs identical values, which the firmware
    # ignores). The run must force a reload so the ghost offset cannot be
    # read as the baseline and ballooned into a hundreds-of-steps fix.
    d = FakeDisplay(total=4)
    d.seed_module_error(1, d.spc)   # real persisted fault: shows the next char
    d.preview(1, -1, -500)          # ghost residue from a prior run
    assert d.res_mod[1] == -500
    d.previews = []                 # only count nudges the run itself makes
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.reloads >= 1           # baseline cleaned before P0
    assert d.res_mod[1] == 0
    assert d.mod_off[1] == 0
    # The correction is one character, not the ~500-step ghost offset.
    assert [delta for _, ci, delta in d.previews if ci < 0] == [d.spc]


def test_unreliable_reads_escalate_without_corrections(tmp_path):
    # A module whose reads disagree with the commanded character across the
    # sweep (frozen reader) has no trustworthy shift mode: the tool must
    # flag it instead of "fixing" it from noise.
    d = FakeDisplay(total=4)
    d.seed_module_error(2, -d.spc)
    reader = SimReader(d)
    reader.frozen[2] = "X"
    calib = VlmCalibrator(d, FakeCamera(), reader, photo_dir=str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full")
    report = calib.run()
    assert report["result"] == "needs-human"
    notes = [e["note"] for e in report["identity"]["persistent"]]
    assert any("unreliable reads" in n or "purity" in n for n in notes)
    assert calib.previews < calib.max_previews


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


def test_dry_run_sweeps_without_touching(tmp_path):
    d = FakeDisplay(total=4)
    d.seed_module_error(1, -d.spc)
    events = []
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="dry-run",
                          on_event=events.append)
    report = calib.run()
    assert report["result"] == "needs-human"
    assert "dry-run" in report["reason"]
    assert d.mod_off[1] == d.spc           # nothing applied
    assert not d.persists
    assert not d.batches
    # The sweep did run: one frame per drum character, shift table logged.
    sweep_reads = [e for e in events
                   if e["text"].startswith("sw_")]
    assert len(sweep_reads) == len(d.drum)
    assert any("shift mode" in e["text"] for e in events)


def test_sweep_covers_every_character_once_in_reverse(tmp_path):
    # P1 measures the whole drum: every character is commanded once on every
    # module, in reverse drum order (a full revolution per step).
    d = FakeDisplay(total=4, charset=48)
    events = []
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full", on_event=events.append)
    calib.run()
    shown = [e["text"].split(":", 1)[0] for e in events
             if e["kind"] == "read" and e["text"].startswith("sw_")]
    assert len(shown) == len(d.drum)
    assert len(set(shown)) == len(d.drum)  # no character re-shown



def test_summary_records_offset_delta(tmp_path):
    d = FakeDisplay(total=4)
    d.seed_module_error(1, -d.spc)
    _, report = run_calib(d, tmp_path)
    m1 = [m for m in report["summary"]["modules"] if m["module"] == 1][0]
    assert m1["offset_delta"] == -d.spc
    assert not m1["persistent_wrong_glyph"]


def test_read_log_line_reports_expected_vs_read(tmp_path):
    # The read event must show the commanded frame, the transcription
    # and every mismatched module — not just the raw reading.
    d = FakeDisplay(total=4)
    d.seed_module_error(1, -d.spc)  # module 1 shows the previous char
    events = []
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full",
                          on_event=events.append)
    calib.run()
    first = next(e for e in events if e["kind"] == "read")
    assert "want" in first["text"] and "saw" in first["text"]
    assert "m1 want" in first["text"]  # the off module is named


def test_fine_identity_that_does_not_fit_escalates(tmp_path):
    # A per-character fault whose exact fix would leave the char cell outside
    # the firmware's ±32 clamp is hardware: escalate, never write clamped.
    d = FakeDisplay(total=4)
    ci = d.drum.index("O")
    d.seed_char_error(2, ci, 30 * d.spc)
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "needs-human"
    notes = [e["note"] for e in report["identity"]["persistent"]]
    assert any("char cell" in n for n in notes)
    assert d.char_off[2].get(ci, 0) == 30 * d.spc  # untouched


def test_remote_whole_drum_plus_trim_commits_both(tmp_path):
    # A remote module can need a whole-character offset AND a sub-pitch
    # boundary trim: the committed absolute value must include both (the
    # whole delta was only previewed on the group, never written there).
    d = FakeDisplay(total=6, groups=2, charset=48)
    d.flap_window = int(round(d.spc * 0.4))
    d.seed_module_error(4, d.spc)  # g2 local1: whole drum one char behind
    d.seed_flap_error(4, d.drum.index("H"), int(round(d.spc * 0.6)))
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.remote_mod[0][1] != 0
    assert d.remote_mech[0][1][d.drum.index("H")] != 0


def test_confusable_pair_never_becomes_a_correction(tmp_path):
    # O and 0 are indistinguishable on the drum: a reader that swaps them
    # must not be "corrected" (a 12-character shift for O/0). Confusable
    # differences are no-information and acceptance skips them.
    class ConfusedReader(SimReader):
        def read(self, jpeg, total, expected="", charset="", drum=""):
            reading = super().read(jpeg, total, expected, charset, drum)
            for m in reading.modules:
                if m.char == "O":
                    m.char = "0"
                elif m.char == "0":
                    m.char = "O"
            return reading

    d = FakeDisplay(total=4, charset=48)
    calib = VlmCalibrator(d, FakeCamera(), ConfusedReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full")
    report = calib.run()
    assert report["result"] == "converged"
    assert d.mod_off == [0] * d.local
    assert not d.persists



def test_cell_ladder_rejects_non_improving_nudge(tmp_path):
    # Regression: a correct-char/half-flap target scores 2 at baseline, so a
    # candidate that leaves it equally half (score 2) must NOT persist a
    # no-op offset. Only a candidate that beats the baseline may commit.
    d = FakeDisplay(total=4, charset=48)
    d.flap_window = 17
    d.seed_flap_error(0, d.drum.index("E"), 20)  # half, needs centring
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full")
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    plans = [{"module": 0, "group": 1, "local": 0, "char_index": -1,
              "steps": [-1], "absolute": True, "cap": d.spc,
              "targets": ["E"], "guards": ["A"],
              "state": 0, "best": 0, "best_score": 0, "label": "test"}]
    assert calib._cell_ladder(plans) == set()
    assert plans[0]["best"] == 0
    assert not d.persists
    assert d.condition(0, "E") == "half"


def test_report_written_to_disk(tmp_path):
    d = FakeDisplay(total=4)
    run_calib(d, tmp_path)
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "snapshot.json").is_file()
