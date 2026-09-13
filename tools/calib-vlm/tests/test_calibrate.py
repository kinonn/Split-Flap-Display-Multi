"""VlmCalibrator tests: delta math, convergence, escalation, fleet."""

import math

from calib_vlm.calibrate import VlmCalibrator

from tests.fixtures import FakeCamera, FakeDisplay, SimReader


def run_calib(display, tmp_path, mode="full", exhaustive=False, phases=None):
    kwargs = {}
    if phases is not None:
        kwargs["phases"] = phases
    calib = VlmCalibrator(display, FakeCamera(), SimReader(display),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, exhaustive=exhaustive, mode=mode,
                          **kwargs)
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
                          mode="full", skip_enabled=False)
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
                          mode="full", on_event=events.append,
                          skip_enabled=False)
    calib.run()
    shown = [e["text"].split(":", 1)[0] for e in events
             if e["kind"] == "read" and e["text"].startswith("sw_")]
    assert len(shown) == len(d.drum)
    assert len(set(shown)) == len(d.drum)  # no character re-shown


def test_skip_list_excludes_default_punctuation(tmp_path):
    # Default exclusions (".", "'", "-") are never shown: sweep, ladder
    # frames, P2, P4 and acceptance all skip them. Enabled by default.
    d = FakeDisplay(total=4, charset=48)
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full")
    assert calib.skip_enabled is True
    assert set(calib.skip_chars) == {".", "'", "-"}
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    readings, chars = calib._sweep()
    assert "." not in chars and "'" not in chars and "-" not in chars
    assert len(chars) == len(d.drum) - 3
    assert set(readings[0]) == set(chars)


def test_skip_list_disabled_covers_full_drum(tmp_path):
    # Exclusions off: the sweep covers every drum character again.
    d = FakeDisplay(total=4, charset=48)
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full", skip_enabled=False)
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    _, chars = calib._sweep()
    assert len(chars) == len(d.drum)


def test_skip_list_custom_and_ladder_filler_avoids_skipped(tmp_path):
    # A user list replaces the default; ladder background fillers and
    # guard fallbacks never use skipped glyphs either.
    d = FakeDisplay(total=4, charset=48)
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full", skip_chars="AB")
    assert set(calib.skip_chars) == {"A", "B"}
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    _, chars = calib._sweep()
    assert "A" not in chars and "B" not in chars
    plans = [{"module": 0, "group": 1, "local": 0, "char_index": -1,
              "steps": [4], "absolute": True, "cap": d.spc,
              "targets": ["E"], "guards": ["H"],
              "state": 0, "best": 0, "best_score": 0, "label": "test"}]
    calib._cell_ladder(plans)
    shown = [e["frame"] for e in calib.frames
             if e["tag"].startswith("ladder_")]
    assert shown
    for frame in shown:
        # Module 3 is background in every ladder frame.
        assert frame[3] not in {"A", "B"}



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


def test_ladder_filler_rotates_background_slots(tmp_path):
    # Regression (run-003 M0): a constant "E" filler stalls a misaligned
    # background module — target "E" physically showing "F" resolves to
    # identical step positions, the firmware commands zero steps, and
    # every re-read scores the same stale flap (30/30 "E"-want/"F"-saw).
    # Background slots must change glyph on every consecutive ladder
    # frame, on every module, so each show physically moves the drum.
    d = FakeDisplay(total=4, charset=48)
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full")
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    plans = [{"module": 0, "group": 1, "local": 0, "char_index": -1,
              "steps": [4, -4], "absolute": True, "cap": d.spc,
              "targets": ["E"], "guards": ["A", "H"],
              "state": 0, "best": 0, "best_score": 0, "label": "test"}]
    calib._cell_ladder(plans)
    shown = [e["frame"] for e in calib.frames
             if e["tag"].startswith("ladder_")]
    assert len(shown) >= 2
    for prev, cur in zip(shown, shown[1:]):
        # Module 3 is background in every frame: must always move.
        assert prev[3] != cur[3], f"background stalled: {prev!r} -> {cur!r}"


def test_ladder_skips_reshow_when_nothing_moved(tmp_path):
    # A plan cell whose state did not change keeps its cached score:
    # no new frames/VLM calls are burned re-reading it.
    d = FakeDisplay(total=4, charset=48)
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full")
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    plans = [{"module": 0, "group": 1, "local": 0, "char_index": -1,
              "steps": [9999], "absolute": True, "cap": d.spc,
              "targets": ["E"], "guards": ["A"],
              "state": 0, "best": 0, "best_score": 0, "label": "test"}]
    # 9999 exceeds the cap, so the candidate collapses to best (no move).
    before = calib.frames_used
    calib._cell_ladder(plans)
    ladder_frames = [e for e in calib.frames
                     if e["tag"].startswith("ladder_")]
    # Baseline only (1 target + 1 guard); the capped step adds nothing.
    assert len(ladder_frames) == 2, [e["tag"] for e in ladder_frames]
    assert calib.frames_used == before + 2


def test_majority_residual_routes_to_module_fix(tmp_path):
    # run-005 regression: a module whose reads are +1 on ~70% of the drum
    # misses the 80% purity gate and was escalated as "unreliable"; every
    # residual then dead-ended in P2 ("identity fix does not fit a char
    # cell" — a whole character can never fit the ±32 clamp). A >=50%
    # plurality single-char residual is fixed on the module cell instead.
    d = FakeDisplay(total=4, charset=48)
    d.seed_module_error(0, d.spc)  # whole drum one character ahead
    # Cancel a minority (~30%) of characters so purity lands ~70%.
    minority = list(d.drum[1:15])
    for ch in minority:
        d.seed_char_error(0, d.drum.index(ch), -d.spc)
    events = []
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full", on_event=events.append,
                          skip_enabled=False)
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    calib._p1_coarse()
    # Module cell committed back to zero (fault removed), no unreliable
    # escalation, and the majority route is named in the log.
    assert d.mod_off[0] == 0, d.mod_off
    assert not any("unreliable reads" in e["text"] for e in events)
    assert any("majority shift +1" in e["text"] for e in events)


def test_scattered_residuals_still_escalate(tmp_path):
    # Without a majority residual, below-purity reads remain reader noise:
    # escalate, never "fix" from a weak plurality.
    d = FakeDisplay(total=4, charset=48)
    for ch in list(d.drum[1:15]):
        d.seed_char_error(0, d.drum.index(ch), d.spc)
    for ch in list(d.drum[15:24]):
        d.seed_char_error(0, d.drum.index(ch), -d.spc)
    events = []
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full", on_event=events.append,
                          skip_enabled=False)
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    calib._p1_coarse()
    assert any("unreliable reads" in e["text"] for e in events)
    assert not any("majority shift" in e["text"] for e in events)
    assert d.mod_off[0] == 0  # nothing applied


def test_blank_read_against_nonblank_command_is_junk(tmp_path):
    # run-005: the ':' frame read blank fleet-wide, poisoning every module's
    # shift histogram with a phantom ~+10 residual and dragging purities
    # below the trust gate. A blank read against a non-blank command is a
    # failed sampling frame -> excluded (None), not a residual.
    d = FakeDisplay(total=4, charset=48)
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full")
    calib.total, calib.drum = d.total, d.drum
    from calib_vlm.reader import ModuleReading
    blank_against_colon = ModuleReading(0, " ", "blank", 0.9, "vlm", ":")
    assert calib._shift(blank_against_colon, ":") is None
    blank_against_blank = ModuleReading(0, " ", "clean", 0.9, "vlm", " ")
    assert calib._shift(blank_against_blank, " ") == 0


def test_single_flap_arc_probes_proportionate_candidate_no_fix(tmp_path):
    # run-005 hypothesis, arc of FULL-flap-clean errors (reads the next
    # flap cleanly): the module enters the trim ladder and the
    # proportionate candidate round(arc/48 * stepsPerChar) IS probed, but
    # no sub-pitch shift can centre a full-flap fault without breaking the
    # correct majority — so nothing is committed and P2 flags the arc.
    d = FakeDisplay(total=4, charset=48)
    d.flap_window = int(round(d.spc * 0.4))
    arc = list(d.drum[1:14])  # 13 chars = 27% of the drum
    for ch in arc:
        d.seed_char_error(0, d.drum.index(ch), d.spc)  # full flap, clean
    events = []
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full", on_event=events.append,
                          skip_enabled=False)
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    calib._p1_coarse()
    assert any("single-flap arc" in e["text"] for e in events)
    assert not any("unreliable reads" in e["text"] for e in events)
    # The proportionate candidate (round(13/48 * 43) = 12) was probed.
    assert any(delta == 12 for _, ci, delta in d.previews if ci < 0)
    # ...but rejected by the score: nothing committed, no persists.
    assert d.mod_off[0] == 0
    assert not d.persists


def test_single_flap_arc_boundary_flap_commits_trim(tmp_path):
    # When the arc's landings sit part-way past the flap boundary, the
    # proportionate candidate (or the ladder's fine search) CAN pull them
    # back into the readable window without breaking the guards — then
    # and only then is the module offset committed.
    d = FakeDisplay(total=4, charset=48)
    d.flap_window = int(round(d.spc * 0.4))
    arc = list(d.drum[1:14])  # 13 chars = 27% of the drum
    for ch in arc:
        # +0.72 flap: reads the NEXT character; a proportional shift can
        # pull these back inside the readable window.
        d.seed_char_error(0, d.drum.index(ch), int(round(d.spc * 0.72)))
    events = []
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full", on_event=events.append,
                          skip_enabled=False)
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    calib._p1_coarse()
    assert any("single-flap arc" in e["text"] for e in events)
    # The ladder found a shift that cleans the arc without breaking the
    # guards: committed on the module cell.
    assert d.mod_off[0] != 0, d.mod_off
    assert d.persists


def test_ladder_exhausted_guards_rotate_and_rescore(tmp_path):
    # A plan with fewer guards than guards_at has exhausted slots that
    # rotate with the frame counter: the drum physically moves there, so
    # a non-nudged plan must be scored FRESH each evaluate (the glyph
    # sequence changed), never from the cached score.
    d = FakeDisplay(total=4, charset=48)
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full")
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    plans = [
        {"module": 0, "group": 1, "local": 0, "char_index": -1,
         "steps": [4, -4], "absolute": True, "cap": d.spc,
         "targets": ["E"], "guards": ["A", "H", "M"],
         "state": 0, "best": 0, "best_score": 0, "label": "mover"},
        {"module": 1, "group": 1, "local": 1, "char_index": -1,
         "steps": [0], "absolute": True, "cap": d.spc,
         "targets": ["E"], "guards": ["A", "H"],
         "state": 0, "best": 0, "best_score": 0, "label": "static"},
    ]
    calib._cell_ladder(plans)
    g2 = [e["frame"] for e in calib.frames if e["tag"] == "ladder_g2"]
    # Baseline + 2 steps, one g2 frame each.
    assert len(g2) == 3, [e["tag"] for e in calib.frames]
    # Module 1's exhausted slot (guard index 2) must rotate between
    # evaluates instead of pinning guards[0].
    assert g2[0][1] != g2[1][1], g2
    assert g2[1][1] != g2[2][1], g2


def test_report_written_to_disk(tmp_path):
    d = FakeDisplay(total=4)
    run_calib(d, tmp_path)
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "snapshot.json").is_file()


def test_aborts_after_too_many_unreliable_reads(tmp_path):
    # run-001 burned 53 min / the preview budget on a blind camera:
    # 4 unreliable P1 modules then ~124 unreadable P2 escalations.
    # More than MAX_UNRELIABLE_READS reader-trust escalations must stop
    # the run immediately instead of tuning noise.
    import pytest
    from calib.display import CalibError
    from calib_vlm.calibrate import MAX_UNRELIABLE_READS
    d = FakeDisplay(total=4)
    events = []
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full",
                          on_event=events.append)
    for i in range(MAX_UNRELIABLE_READS):
        calib._escalate(0, "?", f"unreliable reads ({i} samples)")
    assert calib._unreliable_count == MAX_UNRELIABLE_READS
    # The (MAX+1)th escalation raises: the run aborts via CalibError.
    with pytest.raises(CalibError, match="too many unreliable reads"):
        calib._escalate(0, "?", "unreadable during fine pass")


def test_unreadable_escalation_counts_toward_abort(tmp_path):
    from calib_vlm.calibrate import MAX_UNRELIABLE_READS
    d = FakeDisplay(total=4)
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full")
    for _ in range(MAX_UNRELIABLE_READS):
        calib._escalate(1, "X", "unreadable during fine pass")
    assert calib._unreliable_count == MAX_UNRELIABLE_READS
    # Non-reader escalations (char-cell clamp) must NOT count.
    calib._escalate(1, "X", "identity fix does not fit a char cell")
    assert calib._unreliable_count == MAX_UNRELIABLE_READS


def test_display_api_calls_logged(tmp_path):
    # Every mutating display call must appear in the log as an `api`
    # event; status polls stay unlogged (wait_settled polls every 0.5 s).
    d = FakeDisplay(total=4)
    events = []
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full",
                          on_event=events.append)
    calib.total, calib.charset, calib.drum = d.total, d.charset, d.drum
    calib.group_widths = [d.total]
    calib.steps_per_char = d.spc
    calib.display.show_and_settle("AB  ", 0, 5)
    calib.display.preview(0, -1, 4)
    calib.display.persist(1, "module", 4, 0, 0)
    kinds = [e["kind"] for e in events]
    assert "api" in kinds
    api_texts = [e["text"] for e in events if e["kind"] == "api"]
    assert any("POST show" in t for t in api_texts)
    assert any("POST preview" in t for t in api_texts)
    assert any("POST offsets" in t for t in api_texts)


def test_vlm_call_logged_with_model_and_timing(tmp_path):
    # Each frame read must log one `vlm` event with model + round trips.
    d = FakeDisplay(total=4)
    events = []
    reader = SimReader(d)
    reader.vlm = type("V", (), {"model": "test-model"})()
    reader.last_calls = 1
    calib = VlmCalibrator(d, FakeCamera(), reader,
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full",
                          on_event=events.append)
    calib.total, calib.charset, calib.drum = d.total, d.charset, d.drum
    calib.group_widths = [d.total]
    calib.steps_per_char = d.spc
    calib._show_read("AB  ", "probe")
    vlm_events = [e for e in events if e["kind"] == "vlm"]
    assert len(vlm_events) == 1
    assert "test-model" in vlm_events[0]["text"]
    assert "probe" in vlm_events[0]["text"]


def test_events_carry_elapsed_seconds(tmp_path):
    # Post-run speed analysis needs sub-second timing: every event must
    # carry monotonic seconds since run start (wall `t` is 1 s resolution).
    d = FakeDisplay(total=4)
    events = []
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full",
                          on_event=events.append)
    calib.event("phase", "probe phase")
    assert events[0]["elapsed"] >= 0.0
    assert isinstance(events[0]["elapsed"], float)


def test_api_events_carry_duration(tmp_path):
    # Per-call display cost (homing dominates) must be visible in the log.
    d = FakeDisplay(total=4)
    events = []
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full",
                          on_event=events.append)
    calib.total, calib.charset, calib.drum = d.total, d.charset, d.drum
    calib.group_widths = [d.total]
    calib.steps_per_char = d.spc
    calib.display.show_and_settle("AB  ", 0, 5)
    api_texts = [e["text"] for e in events if e["kind"] == "api"]
    assert api_texts and all(" in " in t and t.endswith("s") for t in api_texts)


def test_report_has_timing_context_and_traceback(tmp_path):
    # Root-cause + speed analysis from report.json alone: run config,
    # firmware identity, wall/VLM seconds, per-phase costs, crash trace.
    d = FakeDisplay(total=4)
    d.seed_module_error(1, -d.spc)
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert report["config"]["dwell_ms"] == 0
    assert report["config"]["stepsPerRot"] == 2048
    assert report["firmware"]["contractVersion"] == 1
    assert report["timing"]["wallSeconds"] >= 0
    assert report["timing"]["vlmSeconds"] >= 0
    assert report["timing"]["vlmTokens"]["prompt_tokens"] >= 0
    phases = [p["phase"] for p in report["timing"]["phases"]]
    assert any("P1 coarse" in p for p in phases)
    assert all(set(p) >= {"phase", "seconds", "frames", "vlmCalls",
                          "previews", "persists"} for p in report["timing"]["phases"])


def test_phases_reject_unknown_names(tmp_path):
    d = FakeDisplay(total=4)
    try:
        VlmCalibrator(d, FakeCamera(), SimReader(d),
                      photo_dir=str(tmp_path), phases=["p1", "nope"])
    except ValueError as exc:
        assert "unknown phases" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown phase")


def test_p1_only_skips_later_phases(tmp_path):
    # P1-only: module offsets commit, P2/P4/acceptance never run, and the
    # report marks them skipped with a subset verdict.
    d = FakeDisplay(total=4)
    d.seed_module_error(1, -d.spc)  # module 1 shows the previous char
    calib, report = run_calib(d, tmp_path, phases=["p1"])
    assert report["phases"] == ["p1"]
    assert report["skipped"] == ["p2", "p4", "acceptance"]
    assert d.mod_off[1] == 0
    assert d.persists
    assert report["result"] == "needs-human"
    assert "acceptance skipped" in report["reason"]
    assert "P2 fine" not in str(report["timing"]["phases"])
    assert "acceptance" not in str(report["timing"]["phases"]).lower() or \
        "acceptance skipped" in report["reason"]


def test_p2_without_p1_derives_flagged_readonly(tmp_path):
    # P2 without P1: a read-only sweep derives the flagged chars (no
    # module commits), then P2 tunes the per-char cell.
    d = FakeDisplay(total=4)
    ci = d.drum.index("O")
    d.seed_char_error(2, ci, -d.spc)  # only 'O' is one char behind on m2
    calib, report = run_calib(d, tmp_path, phases=["p2"])
    assert report["phases"] == ["p2"]
    assert report["skipped"] == ["p1", "p4", "acceptance"]
    assert d.char_off[2].get(ci, 0) == 0
    # No module-cell writes: the read-only sweep must not commit.
    assert all(x["charIndex"] >= 0 for x in report["deltas"])
    assert d.mod_off == [0] * d.local


def test_p4_only_is_readonly(tmp_path):
    # P4/acceptance-only: no previews or persists, verdict escalates.
    d = FakeDisplay(total=4)
    calib, report = run_calib(d, tmp_path, phases=["p4"])
    assert report["phases"] == ["p4"]
    assert report["skipped"] == ["p1", "p2", "acceptance"]
    assert d.previews == []
    assert d.persists == []
    assert report["result"] == "needs-human"
