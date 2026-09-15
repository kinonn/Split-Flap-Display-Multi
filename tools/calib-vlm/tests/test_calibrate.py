"""VlmCalibrator tests: delta math, convergence, escalation, fleet."""

import math

import pytest

from calib.display import CalibError
from calib_vlm.calibrate import (BATCH_MAX_NUDGES, REMOTE_BATCH_MAX_NUDGES,
                                 VlmCalibrator, _p2_ladder_steps)

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
    # Only 'O' is one char behind on m2, with the cell at the clamp edge:
    # the incremental ladder must walk it back inside ±32 in one step.
    d.seed_char_error(2, ci, -32)
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.char_off[2].get(ci, 0) == 0
    # A single-glyph fault must stay on a char cell; a coarse module shift
    # would break every other character on that drum.
    m2 = [x for x in report["deltas"] if x["globalModule"] == 2]
    assert m2 and all(x["charIndex"] >= 0 for x in m2)
    assert all(abs(delta) <= 32 for _, ci, delta in d.previews if ci >= 0)


def test_per_char_identity_overflow_escalates_cleanly(tmp_path):
    # A per-character fault that needs a whole flap (a 37-char drum 30
    # positions away is +7 chars = 385 motor steps here) cannot be reached
    # by any candidate inside the firmware's ±32 char-cell clamp: the
    # incremental ladder probes first, then escalates reporting the probes
    # - it never writes a clamped, wrong offset that would leave the
    # display worse than before.
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
    calib._p1_flagged = ["O"]
    calib._p2_fine()
    assert d.char_off[2].get(ci, 0) == 0   # cell untouched, not clamped
    assert d.previews                      # ...but the ladder did probe
    assert not d.persists
    notes = [e["note"] for e in calib.identity_persistent]
    assert any("incremental ladder" in n and "char cell" in n for n in notes)


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
    calib._p1_flagged = ["H"]
    calib._p2_fine()
    assert d.remote_char[0][2][ci] == 0    # not corrupted to ±32
    assert d.batches                       # remote ladder probes were sent
    assert not d.previews and not d.persists
    notes = [e["note"] for e in calib.identity_persistent]
    assert any("incremental ladder" in n and "char cell" in n for n in notes)


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


def test_batch_nudge_chunks_to_the_firmware_caps(tmp_path):
    # kinonn-bot#40/#38: _batch_nudge sent every nudge of one scope in ONE
    # preview-batch call. The firmware rejects >48 nudges per call (HTTP
    # 400, not retryable) and its loop-task drain forwards only 8 nudges
    # per remote group, silently dropping the rest — so a large trim plan
    # either aborted the run or lost nudges. The fixture display enforces
    # both real caps, so this test fails loudly on the old code.
    d = FakeDisplay(total=18, groups=3, charset=48)
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full")
    calib.total, calib.charset, calib.drum = d.total, d.charset, d.drum
    calib.group_widths = d.widths()
    calib.steps_per_char = d.spc
    calib._load_remote_offsets(d.snapshot()["settings"])
    nudges = ([(1, m, ci, 1) for m in range(6) for ci in range(10)]      # 60
              + [(2, m, ci, 1) for m in range(2) for ci in range(10)]     # 20
              + [(3, m, ci, 1) for m in range(2) for ci in range(10)])    # 20
    calib._batch_nudge(nudges)
    # No >48 call (HTTP 400) and no silent remote truncation: the batches
    # respect the caps and every nudge landed on the device.
    for batch in d.batches:
        assert len(batch) <= BATCH_MAX_NUDGES, len(batch)
        for scope in {n["scope"] for n in batch if n["scope"] >= 2}:
            count = sum(1 for n in batch if n["scope"] == scope)
            assert count <= REMOTE_BATCH_MAX_NUDGES, (scope, count)
    want: dict[tuple[int, int, int], int] = {}
    for group, module, ci, delta in nudges:
        want[(group, module, ci)] = want.get((group, module, ci), 0) + delta
    for (group, module, ci), delta in want.items():
        if group == 1:
            assert d.res_char[module][ci] == delta
        else:
            assert d.res_remote_char[group - 2][module][ci] == delta
    assert calib.previews == len(nudges)
    # 60 local nudges -> two <=48 calls; each 20-nudge remote scope -> three
    # <=8 calls. Nothing is re-sent or reordered.
    assert len(d.batches) == 2 + 3 + 3
    flat = [n for batch in d.batches for n in batch]
    assert [(n["scope"], n["module"], n["charIndex"], n["delta"])
            for n in flat] == sorted(nudges), "nudges lost or reordered"


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
    # shift: the module trim must not touch that module (later phases may
    # still walk the individual char cells).
    d = FakeDisplay(total=4, charset=48)
    d.flap_window = int(round(d.spc * 0.4))
    d.seed_flap_error(1, d.drum.index("H"), int(round(d.spc * 0.6)))
    d.seed_flap_error(1, d.drum.index("D"), -int(round(d.spc * 0.6)))
    calib, report = run_calib(d, tmp_path)
    assert all(not (n["module"] == 1 and n["charIndex"] < 0)
               for batch in d.batches for n in batch)


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


def test_ahead_by_one_char_converges_within_clamp(tmp_path):
    # The aborted-run regression: commanded E, showed F (one char AHEAD)
    # with the cell at the clamp edge. The incremental ladder walks it back
    # with a single <=32-step candidate, never a near-full-revolution
    # forward grind and never a >32-step jump.
    d = FakeDisplay(total=4)
    ci = d.drum.index("E")
    d.seed_char_error(1, ci, 32)  # 'E' shows the NEXT char on m1
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.char_off[1].get(ci, 0) == 0
    char_previews = [(c, d_) for _, c, d_ in d.previews if c >= 0]
    assert all(abs(delta) <= 32 for _, delta in char_previews)


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


def test_fleet_geometry_uses_declared_group_widths(tmp_path):
    # kinonn-bot#41: the firmware maps fleet modules through the
    # user-editable masterGroupModuleCounts (group 1 first), not through
    # local-wide groups. With local 8 and counts 8,6,4 (total 18) the old
    # heuristic derived [8,8,2], so fleet module 14 - group 3, local 0 -
    # was tuned as group 2 local 6 and never fixed.
    d = FakeDisplay(group_widths=[8, 6, 4])
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full")
    status, settings = d.status(), d.snapshot()["settings"]
    assert (status["numModules"], status["totalModules"],
            status["groupCount"]) == (8, 18, 3)
    assert settings["masterGroupModuleCounts"] == "8,6,4"
    assert calib._widths(status, settings) == [8, 6, 4]
    calib.group_widths = [8, 6, 4]
    assert (calib._group_of(7), calib._local_index(7)) == (1, 7)
    assert (calib._group_of(8), calib._local_index(8)) == (2, 0)
    assert (calib._group_of(13), calib._local_index(13)) == (2, 5)
    assert (calib._group_of(14), calib._local_index(14)) == (3, 0)
    assert (calib._group_of(17), calib._local_index(17)) == (3, 3)


def test_fleet_run_tunes_the_declared_mapping(tmp_path):
    # End to end: a fault on fleet module 14 is a group-3 fault, and the
    # report carries the geometry the run actually used.
    d = FakeDisplay(group_widths=[8, 6, 4])
    d.seed_module_error(14, -d.spc)  # group 3, local module 0
    calib, report = run_calib(d, tmp_path)
    assert report["fleet"]["groupWidths"] == [8, 6, 4]
    assert report["result"] == "converged"
    assert d.remote_mod[2][0] == 0
    assert all(not row for row in d.remote_mod[1])  # group 2 untouched


def test_fleet_geometry_prefers_status_group_widths(tmp_path):
    # The status endpoint's groupWidths (new firmware field) wins over a
    # stale /settings CSV, and a present-but-inconsistent field is an
    # error - guessing a mapping silently mis-tunes every later group.
    d = FakeDisplay(group_widths=[8, 6, 4], master_counts="8,8,8,8,8,8",
                    status_group_widths=[8, 6, 4])
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full")
    assert calib._widths(d.status(), d.snapshot()["settings"]) == [8, 6, 4]
    d.declared_status_widths = [8, 8, 8]  # sums to 24, not 18
    with pytest.raises(CalibError, match="fleet geometry inconsistent"):
        calib._widths(d.status(), d.snapshot()["settings"])
    d.declared_status_widths = [8, 6]     # three groups, two widths
    with pytest.raises(CalibError, match="fleet geometry inconsistent"):
        calib._widths(d.status(), d.snapshot()["settings"])


def test_fleet_geometry_falls_back_to_equal_width_heuristic(tmp_path):
    # Firmware that declares no widths keeps the legacy equal-width layout
    # (last group short, so an 8-module/3-group display is 2,2,4).
    d = FakeDisplay(total=8, groups=3, master_counts="")
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full")
    assert "groupWidths" not in d.status()
    assert d.snapshot()["settings"]["masterGroupModuleCounts"] == ""
    assert calib._widths(d.status(), d.snapshot()["settings"]) == [2, 2, 4]
    # A stale CSV that does not add up is ignored the same way.
    stale = FakeDisplay(total=8, groups=3, master_counts="8,8,8")
    assert calib._widths(stale.status(), stale.snapshot()["settings"]) == \
        [2, 2, 4]


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


def test_sweep_budget_counts_passes_and_fires(tmp_path):
    # kinonn-bot#48: nothing incremented `self.sweeps`, so the "sweep
    # budget exhausted" guard was inert. One sweep pass = one sweep.
    d = FakeDisplay(total=4)
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full")
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    calib.max_sweeps = 2
    calib._sweep()
    assert calib.sweeps == 1
    calib._sweep()
    assert calib.sweeps == 2
    with pytest.raises(CalibError, match="sweep budget exhausted"):
        calib._sweep()
    assert calib.sweeps == 2   # the rejected pass is not counted


def test_run_enforces_the_sweep_budget(tmp_path):
    # The budget stops the run with a reason instead of grinding through
    # one more whole-drum sweep.
    d = FakeDisplay(total=4)
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full")
    calib.max_sweeps = 0
    report = calib.run()
    assert report["result"] == "needs-human"
    assert "sweep budget exhausted" in report["reason"]


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
    # A per-character fault that needs a whole flap is unreachable from any
    # candidate inside the firmware's ±32 char-cell clamp: the ladder probes
    # it, finds no clean landing, and escalates - never written clamped.
    d = FakeDisplay(total=4)
    ci = d.drum.index("O")
    d.seed_char_error(2, ci, 30 * d.spc)
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "needs-human"
    notes = [e["note"] for e in report["identity"]["persistent"]]
    assert any("incremental ladder" in n and "char cell" in n for n in notes)
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


def test_remote_cell_second_commit_keeps_the_first(tmp_path):
    # kinonn-bot#39: a remote cell is committed twice in one run (P1 runs
    # the module trim and the phase trim over overlapping modules and
    # unions them as handled = trimmed | phased). The second commit
    # recomputed its base from the run-start /settings snapshot, which is
    # never refreshed, so it silently dropped the first verified
    # component. Invariant: the persisted value is the value the device
    # actually holds.
    d = FakeDisplay(total=6, groups=2, charset=48)
    d.flap_window = int(round(d.spc * 0.4))
    d.seed_flap_error(4, d.drum.index("H"), int(round(d.spc * 0.6)))  # g2 local1
    d.seed_flap_error(4, d.drum.index("N"), int(round(d.spc * 1.0)))
    d.seed_flap_error(4, d.drum.index("G"), int(round(d.spc * 1.0)))
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full")
    calib.total, calib.charset, calib.drum = d.total, d.charset, d.drum
    calib.group_widths = d.widths()
    calib.steps_per_char = d.spc
    calib._load_remote_offsets(d.snapshot()["settings"])
    steps = [int(round(d.spc * f)) for f in (0.5, 0.25, 0.125, 0.0625)]

    def plan(label, target, guard):
        return {"module": 4, "group": 2, "local": 1, "char_index": -1,
                "steps": list(steps), "cap": d.spc, "targets": [target],
                "guards": [guard], "state": 0, "best": 0, "best_score": 0,
                "label": label}

    first = plan("trim", "H", "A")
    assert calib._cell_ladder([first]) == {4}
    written = d.remote_mod[0][1]
    assert written and d.displayed_char(4, "H") == "H"
    second = plan("phase", "N", "G")
    assert calib._cell_ladder([second]) == {4}
    assert second["best"] != 0
    # The second commit persists the tracked absolute = first + second,
    # not the run-start snapshot + second.
    assert d.remote_mod[0][1] == written + second["best"]
    assert calib.live((2, 1, -1)) == d.remote_mod[0][1]
    assert [p for p in d.persists if p[0] == 2] == [
        (2, "module", written, 1, 0),
        (2, "module", written + second["best"], 1, 0)]
    # Both components are live on the device: H (first commit) and N
    # (second commit) read clean.
    assert d.displayed_char(4, "H") == "H" and d.condition(4, "H") == "clean"
    assert d.displayed_char(4, "N") == "N" and d.condition(4, "N") == "clean"


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


def test_p2_ladder_scans_increments_up_to_half_a_character(tmp_path):
    # P2's seam ladder scans candidate offsets in steps of 4 motor steps up
    # to half a character pitch, so a fault needing more than the old
    # +/-1/2/4/8 ladder's ceiling is still found. A mechanical landing
    # error of 16 steps with a +/-4 step clean window is cleared only by an
    # offset in [-20, -12]; -12 is the smallest increment the scan offers.
    d = FakeDisplay(total=4, charset=48)
    d.flap_window = 4
    ci = d.drum.index("E")
    d.seed_flap_error(0, ci, 16)
    events = []
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full", on_event=events.append)
    report = calib.run()
    assert report["result"] == "converged"
    assert d.char_off[0].get(ci, 0) == -12
    assert d.condition(0, "E") == "clean"
    assert any("cell ladder" in e["text"] and "-12 steps" in e["text"]
               for e in events)


def test_p2_ladder_stops_probing_a_clean_cell(tmp_path):
    # An accepted candidate that leaves the flap clean is final: every
    # later candidate could only tie it, so the cell must stop being
    # nudged instead of burning the preview budget on the rest of the
    # scan.
    d = FakeDisplay(total=4, charset=48)
    d.flap_window = 17
    ci = d.drum.index("E")
    d.seed_flap_error(0, ci, -18)  # half at the base, clean at +4
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.char_off[0].get(ci, 0) == 4
    # Exactly one candidate was applied; nothing was probed after it.
    assert d.previews == [(0, ci, 4)]
    ladder = [e for e in calib.frames if e["tag"].startswith("ladder_")]
    assert len(ladder) == 8  # baseline + one candidate round (1+3 frames)
    assert d.persists == [(1, "char", 4, 0, ci)]


def test_p2_ladder_steps_default_and_explicit_cap():
    # The seam default caps at half a character pitch; identity faults pass
    # the firmware's ±32 char-cell clamp so the scan can reach the edge.
    steps = _p2_ladder_steps(43)
    assert max(abs(s) for s in steps) == 20  # half-pitch cap (43 // 2 = 21)
    assert steps[-4:] == [2, -2, 1, -1]      # narrow-window fallback
    wide = _p2_ladder_steps(43, cap=32)
    assert wide[:4] == [4, -4, 8, -8]
    assert max(abs(s) for s in wide) == 32
    assert wide[-4:] == [2, -2, 1, -1]


def test_p2_ladder_respects_char_cell_clamp(tmp_path):
    # Char-cell candidates are bounded by the firmware's ±32 clamp: no probe
    # ever pushes the cell's live value outside the window, and the wider
    # identity cap reaches a landing past +4 that the half-pitch seam cap
    # could not (this fault is clean only at +12).
    d = FakeDisplay(total=4, charset=48)
    d.flap_window = 4
    ci = d.drum.index("E")
    d.seed_char_error(0, ci, 15)   # base +15 (already off-centre)
    d.seed_flap_error(0, ci, -30)  # reads 'E', half; clean at a +12 nudge
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full")
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    calib._p1_flagged = ["E"]
    calib._p2_fine()
    assert d.char_off[0].get(ci, 0) == 27  # base 15 + the accepted +12
    assert d.condition(0, "E") == "clean"
    # Replaying the applied nudges (previews record applies AND reverts):
    # the cell never leaves the ±32 window.
    value = 15
    peak = 15
    for _, cell, delta in d.previews:
        if cell < 0:
            continue
        value += delta
        peak = max(peak, abs(value))
        assert -32 <= value <= 32
    assert peak > 15  # the wider identity cap is actually used


def test_ladder_stops_before_the_budget_wall_and_commits(tmp_path):
    # A ladder can run out of preview budget with candidates still
    # untested: it must stop probing while a round is still affordable
    # and let the commit loop persist the offsets already verified.
    # Before the guard the unaffordable round ran anyway and _batch_nudge
    # aborted the run with every winner still uncommitted (run-006's P2
    # died on exactly that).
    d = FakeDisplay(total=4, charset=48)
    d.flap_window = 4
    ci = d.drum.index("E")
    d.seed_flap_error(0, ci, -6)   # clean at +4: the first step settles it
    d.seed_flap_error(1, ci, 44)   # reads a flap ahead whatever is tried
    events = []
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                          dwell_ms=0, timeout_s=5, min_confidence=0.5,
                          mode="full", on_event=events.append)
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    # Round 0 costs 3 previews (winner apply + stuck apply/revert); a
    # second round needs more than the 4 allowed, so it must not run.
    calib.max_previews = 4
    steps = _p2_ladder_steps(d.spc)
    cap = max(1, d.spc // 2)
    plans = [
        {"module": 0, "group": 1, "local": 0, "char_index": ci,
         "steps": steps, "absolute": True, "cap": cap,
         "targets": ["E"], "guards": ["A", "H", "M"],
         "state": 0, "best": 0, "best_score": 0, "label": "winner"},
        {"module": 1, "group": 1, "local": 1, "char_index": ci,
         "steps": steps, "absolute": True, "cap": cap,
         "targets": ["E"], "guards": ["A", "H", "M"],
         "state": 0, "best": 0, "best_score": 0, "label": "stuck"},
    ]
    assert calib._cell_ladder(plans) == {0}
    assert d.char_off[0].get(ci, 0) == 4  # the verified winner, committed
    assert d.char_off[1].get(ci, 0) == 0  # nothing invented for the stuck cell
    assert any("stopped before step" in e["text"] for e in events)
    ladder = [e for e in calib.frames if e["tag"].startswith("ladder_")]
    assert len(ladder) == 8  # baseline + the one affordable round


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
    # residual then dead-ended in P2 (a whole-character fix cannot fit the
    # ±32 char-cell clamp). A >=50% plurality single-char residual is fixed
    # on the module cell instead.
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
    # Keep the mover imperfect: a plan that reads perfect at baseline
    # stops being probed (no candidate could beat it), and the rotation
    # check needs a live evaluate on every round.
    d.seed_flap_error(0, d.drum.index("E"), 20)
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
    calib._escalate(1, "X", "incremental ladder found no clean offset "
                            "on the char cell")
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
    # Each row carries the cost of the phase NAMED in it: the last row is
    # the last phase that ran, not the trailing cleanup mark that closes
    # its interval (run-008 reported every cost one row late).
    assert "acceptance" in phases[-1]


def test_phase_timing_labels_each_row_with_its_own_phase(tmp_path):
    # Marks announce the phase they START, so a slice belongs to the
    # earlier mark: a phase's row must carry that phase's own cost, not
    # whatever ran after it (run-008 shipped the table one row late - the
    # sub-pitch ladder's 14 previews and 2 persists sat under "P2 fine",
    # P4's 142 s under the closing cleanup mark).
    d = FakeDisplay(total=4)
    calib = VlmCalibrator(d, FakeCamera(), SimReader(d),
                          photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                          min_confidence=0.5, mode="full")
    calib._phase_marks = [
        {"name": "A", "elapsed": 10.0, "frames": 1, "vlmCalls": 1,
         "previews": 0, "persists": 0},
        {"name": "B", "elapsed": 30.0, "frames": 5, "vlmCalls": 6,
         "previews": 1, "persists": 0},
        {"name": "C", "elapsed": 45.0, "frames": 9, "vlmCalls": 8,
         "previews": 2, "persists": 1},
    ]
    rows = [(r["phase"], r["seconds"], r["frames"], r["persists"])
            for r in calib._phase_timing()]
    assert rows == [("startup", 10.0, 1, 0),
                    ("A", 20.0, 4, 0),
                    ("B", 15.0, 4, 1)]


def test_phases_reject_unknown_names(tmp_path):
    d = FakeDisplay(total=4)
    try:
        VlmCalibrator(d, FakeCamera(), SimReader(d),
                      photo_dir=str(tmp_path), phases=["p1", "nope"])
    except ValueError as exc:
        assert "unknown phases" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown phase")


def test_preview_mode_is_rejected_not_supported(tmp_path):
    # kinonn-bot#48: the old `self.mode == "preview"` branches were
    # unreachable — the calibrator accepts dry-run/full only (and the
    # server folds the legacy "preview" config into "full"), which is why
    # they were removed rather than kept "just in case".
    d = FakeDisplay(total=4)
    with pytest.raises(ValueError, match="mode must be dry-run or full"):
        VlmCalibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                      dwell_ms=0, timeout_s=5, min_confidence=0.5,
                      mode="preview")


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
