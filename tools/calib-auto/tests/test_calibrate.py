"""Calibrator tests: delta math, convergence, escalation, fleet.

Ported from the calib-vlm suite (the loop is the same code) with the
recognizer renamed; SimReader plays both reader backends.
"""

from collections import Counter
from itertools import pairwise

import pytest
from fakes import FakeCamera, FakeDisplay, SimReader

from calib_auto.calibrate import (
    BATCH_MAX_NUDGES,
    CHAR_OFFSET_LIMIT,
    MAX_UNRELIABLE_READS,
    REMOTE_BATCH_MAX_NUDGES,
    Calibrator,
    _p1_direction,
    _p1_module_delta,
    _p2_ladder_steps,
)
from calib_auto.display import CalibError
from calib_auto.reader import ModuleReading, ReaderError


def run_calib(display, tmp_path, mode="full", exhaustive=False, phases=None):
    kwargs = {}
    if phases is not None:
        kwargs["phases"] = phases
    calib = Calibrator(display, FakeCamera(), SimReader(display),
                       photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                       min_confidence=0.5, exhaustive=exhaustive, mode=mode,
                       **kwargs)
    return calib, calib.run()


def test_coarse_identity_converges(tmp_path):
    d = FakeDisplay(total=4)
    d.seed_module_error(1, -d.spc)  # module 1 shows the previous char
    _calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert report["summary"]["ok"]
    assert d.mod_off[1] == 0
    assert d.persists
    # A whole-drum fault is tuned on the coarse module cell.
    m1 = [x for x in report["deltas"] if x["globalModule"] == 1]
    assert any(x["charIndex"] == -1 for x in m1)
    # Char-cell previews respect the firmware's per-call ±32 clamp; module
    # cells are unbounded and applied in one shot.
    assert d.previews
    assert all(abs(delta) <= 32 for _, ci, delta in d.previews if ci >= 0)


def test_per_char_identity_converges(tmp_path):
    d = FakeDisplay(total=4)
    ci = d.drum.index("O")
    # Only 'O' is one char behind on m2, with the cell at the clamp edge:
    # the incremental ladder must walk it back inside ±32 in one step.
    d.seed_char_error(2, ci, -32)
    _calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.displayed_char(2, "O") == "O"
    assert -32 <= d.char_off[2].get(ci, 0) <= 32
    # A single-glyph fault must stay on a char cell; a coarse module shift
    # would break every other character on that drum.
    m2 = [x for x in report["deltas"] if x["globalModule"] == 2]
    assert m2 and all(x["charIndex"] >= 0 for x in m2)
    assert all(abs(delta) <= 32 for _, ci, delta in d.previews if ci >= 0)


def test_per_char_identity_overflow_escalates_cleanly(tmp_path):
    # A per-character fault that needs a whole flap cannot be reached by
    # any candidate inside the firmware's ±32 char-cell clamp: the
    # incremental ladder probes first, then escalates reporting the probes
    # - it never writes a clamped, wrong offset.
    d = FakeDisplay(total=4)
    ci = d.drum.index("O")
    far = d.drum[(ci + 30) % len(d.drum)]
    reader = SimReader(d)
    reader.frozen[2] = far
    calib = Calibrator(d, FakeCamera(), reader, photo_dir=str(tmp_path),
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
    calib = Calibrator(d, FakeCamera(), reader, photo_dir=str(tmp_path),
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


def test_sub_pitch_offset_is_invisible_and_left_alone(tmp_path):
    # This hardware always lands on a full glyph, so a sub-step landing
    # error rounds to the same character, reads clean, and is no fault.
    d = FakeDisplay(total=4)
    ci = d.drum.index("A")
    d.seed_char_error(3, ci, 1)
    _calib, report = run_calib(d, tmp_path, exhaustive=True)
    assert report["result"] == "converged"
    assert d.char_off[3].get(ci, 0) == 1  # untouched


def test_per_char_offsets_may_differ_within_a_module(tmp_path):
    # Each glyph on a module can need its own char-cell offset. One
    # character lands a flap ahead, another a flap behind; both are
    # corrected independently, on the SAME module.
    d = FakeDisplay(total=4, charset=48)
    ahead = d.drum.index("H")
    behind = d.drum.index("D")
    d.seed_char_error(1, ahead, 32)    # 'H' shows the next flap
    d.seed_char_error(1, behind, -32)  # 'D' shows the previous flap
    _calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.displayed_char(1, "H") == "H"
    assert d.displayed_char(1, "D") == "D"
    assert d.char_off[1][ahead] != d.char_off[1][behind]  # independent
    assert d.mod_off[1] == 0  # no module-wide correction
    m1 = [x for x in report["deltas"] if x["globalModule"] == 1]
    assert m1 and all(x["charIndex"] >= 0 for x in m1)


def test_sweep_is_reverse_drum_order(tmp_path):
    # The sweep walks the drum backwards so every step is ~a full
    # revolution (per-frame magnet re-home).
    d = FakeDisplay(total=4, charset=48)
    seen = []
    orig = d.show_and_settle

    def counting(frame, *a, **k):
        seen.append(frame)
        return orig(frame, *a, **k)

    d.show_and_settle = counting
    calib = Calibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                       dwell_ms=0, timeout_s=5, min_confidence=0.5,
                       mode="full")
    calib.total, calib.charset, calib.drum = d.total, d.charset, d.drum
    _readings, chars = calib._sweep()
    assert chars == list(reversed(d.drum))
    assert seen == [ch * d.total for ch in reversed(d.drum)]


def test_char_fixes_apply_in_parallel_batches(tmp_path):
    # Independent character cells are tuned in the same rounds: one batch
    # per round carries every candidate.
    d = FakeDisplay(total=4, charset=48)
    ci = d.drum.index("H")
    d.seed_char_error(1, ci, 32)
    d.seed_char_error(3, ci, 32)
    _calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    multi = [b for b in d.batches if len(b) >= 2]
    assert multi, "expected a batch carrying both cells' candidates"
    assert d.displayed_char(1, "H") == "H"
    assert d.displayed_char(3, "H") == "H"
    assert len(d.batches) <= 24


def test_batch_nudge_chunks_to_the_firmware_caps(tmp_path):
    # _batch_nudge must chunk: the firmware rejects >48 nudges per call
    # (HTTP 400) and its drain forwards only 8 nudges per remote group,
    # silently dropping the rest. The fixture enforces both caps.
    d = FakeDisplay(total=18, groups=3, charset=48)
    calib = Calibrator(d, FakeCamera(), SimReader(d),
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


def test_dry_run_leaves_char_faults_untouched(tmp_path):
    # Dry-run is read-only: even a character fault the full mode would fix
    # must stay untouched (no previews, no persists, no batches).
    d = FakeDisplay(total=4, charset=48)
    ci = d.drum.index("H")
    d.seed_char_error(1, ci, 32)
    _calib, _report = run_calib(d, tmp_path, mode="dry-run")
    assert not d.previews
    assert not d.persists
    assert not d.batches
    assert d.char_off[1].get(ci, 0) == 32


def test_module_fix_clears_char_suspects_without_char_tunes(tmp_path):
    # After a module-cell fix, P2 suspects on that module re-verify clean —
    # no char cell is ever touched.
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
    _calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.remote_mod[0][1] == 0


def test_ahead_by_one_char_converges_within_clamp(tmp_path):
    # The aborted-run regression: commanded E, showed F (one char AHEAD)
    # with the cell at the clamp edge. The incremental ladder walks it
    # back with a single <=32-step candidate.
    d = FakeDisplay(total=4)
    ci = d.drum.index("E")
    d.seed_char_error(1, ci, 32)  # 'E' shows the NEXT char on m1
    _calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.displayed_char(1, "E") == "E"
    assert d.char_off[1].get(ci, 0) == 24  # one -8 step settles it
    char_previews = [(c, d_) for _, c, d_ in d.previews if c >= 0]
    assert all(abs(delta) <= 32 for _, delta in char_previews)
    assert all(c >= 0 for _, c, _ in d.previews)  # no module-cell jump


def test_module_cell_fault_applied_in_one_preview(tmp_path):
    # A whole-drum (module-cell) fault must be corrected with the sign the
    # firmware actually uses and in a SINGLE preview. P1 scales the exact
    # float pitch, so a 12-char fault on the 37-drum corrects by
    # round(2048/37 * -12) = -664 rather than -12 * spc (-660); the
    # 4-step remainder is sub-pitch and reads clean.
    d = FakeDisplay(total=4)
    d.seed_module_error(1, -d.spc * 12)  # every glyph 12 chars off
    _calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.mod_off[1] == -4
    mod_previews = [(ci, delta) for _, ci, delta in d.previews if ci < 0]
    assert len(mod_previews) == 1          # one re-home, not 30+
    assert mod_previews[0][1] == -664


def test_run_reverts_volatile_preview_residue(tmp_path):
    # A previous aborted or preview run leaves RAM-only preview residue
    # that never reverts (the settings rollback re-POSTs identical values,
    # which the firmware ignores). The run must force a reload so the
    # ghost offset cannot be read as the baseline.
    d = FakeDisplay(total=4)
    d.seed_module_error(1, d.spc)   # real persisted fault: shows next char
    d.preview(1, -1, -500)          # ghost residue from a prior run
    assert d.res_mod[1] == -500
    d.previews = []                 # only count nudges the run itself makes
    _calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.reloads >= 1           # baseline cleaned before P0
    assert d.res_mod[1] == 0
    assert d.mod_off[1] == 0
    # The correction is one character, not the ~500-step ghost offset.
    assert [delta for _, ci, delta in d.previews if ci < 0] == [d.spc]


def test_starved_histograms_escalate_without_corrections(tmp_path):
    # A reader that reports no confident glyph leaves every histogram
    # starved (< 24 trusted samples): flag each module, never "fix" noise.
    d = FakeDisplay(total=4)
    d.seed_module_error(2, -d.spc)
    reader = SimReader(d, confidence=0.0)
    calib = Calibrator(d, FakeCamera(), reader, photo_dir=str(tmp_path),
                       dwell_ms=0, timeout_s=5, min_confidence=0.5,
                       mode="full")
    report = calib.run()
    assert report["result"] == "needs-human"
    notes = [e["note"] for e in report["identity"]["persistent"]]
    assert any("unreliable reads" in n for n in notes)
    assert calib.previews == 0
    assert not d.persists
    assert not d.batches


def test_fleet_geometry_uses_declared_group_widths(tmp_path):
    # The firmware maps fleet modules through masterGroupModuleCounts
    # (group 1 first), not through local-wide groups.
    d = FakeDisplay(group_widths=[8, 6, 4])
    calib = Calibrator(d, FakeCamera(), SimReader(d),
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
    _calib, report = run_calib(d, tmp_path)
    assert report["fleet"]["groupWidths"] == [8, 6, 4]
    assert report["result"] == "converged"
    assert d.remote_mod[2][0] == 0
    assert all(not row for row in d.remote_mod[1])  # group 2 untouched


def test_fleet_geometry_prefers_status_group_widths(tmp_path):
    # The status endpoint's groupWidths wins over a stale /settings CSV,
    # and a present-but-inconsistent field is an error.
    d = FakeDisplay(group_widths=[8, 6, 4], master_counts="8,8,8,8,8,8",
                    status_group_widths=[8, 6, 4])
    calib = Calibrator(d, FakeCamera(), SimReader(d),
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
    calib = Calibrator(d, FakeCamera(), SimReader(d),
                       photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                       min_confidence=0.5, mode="full")
    assert "groupWidths" not in d.status()
    assert d.snapshot()["settings"]["masterGroupModuleCounts"] == ""
    assert calib._widths(d.status(), d.snapshot()["settings"]) == [2, 2, 4]
    # A stale CSV that does not add up is ignored the same way.
    stale = FakeDisplay(total=8, groups=3, master_counts="8,8,8")
    assert calib._widths(stale.status(), stale.snapshot()["settings"]) == \
        [2, 2, 4]


def test_fleet_geometry_rejects_non_positive_legacy_width(tmp_path):
    # The legacy equal-width heuristic must not silently invent a
    # zero/negative group width. With fewer modules than groups the local
    # width truncates to 0, which would map every module onto the last
    # group instead of raising the documented "inconsistent geometry"
    # error (`_parse_widths` already rejects a declared width <= 0).
    d = FakeDisplay(total=4, groups=6, master_counts="")
    calib = Calibrator(d, FakeCamera(), SimReader(d),
                       photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                       min_confidence=0.5, mode="full")
    assert "groupWidths" not in d.status()
    assert d.status()["numModules"] == 0
    with pytest.raises(CalibError, match="fleet geometry inconsistent"):
        calib._widths(d.status(), d.snapshot()["settings"])


def test_batch_nudge_charges_only_applied_nudges(tmp_path):
    # A remote scope with no fleet preview endpoint is skipped; the preview
    # budget must be charged for the nudges actually sent, not the whole
    # requested list. The over-charge used to exhaust the budget early on
    # old firmware ("preview budget exhausted" with work left to do).
    d = FakeDisplay(total=6, groups=2)
    d.preview_batch = None  # firmware without /api/calib/preview-batch
    calib = Calibrator(d, FakeCamera(), SimReader(d),
                       photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                       min_confidence=0.5, mode="full")
    calib._batch_nudge([(1, 0, -1, 8), (2, 0, -1, 8)])
    # The local nudge went out serially; the remote one was skipped.
    assert d.previews == [(0, -1, 8)]
    assert calib.previews == 1
    # The skipped cell's tracked belief is untouched too, so a later commit
    # on it cannot build on a nudge that never happened.
    assert calib.residue.get((2, 0, -1), 0) == 0
    assert d.res_remote_mod[0][0] == 0


def test_remote_group_char_converges(tmp_path):
    # A remote character cell is tuned with RAM-only previews, then its
    # verified absolute value is persisted to the master's mirror.
    d = FakeDisplay(total=6, groups=2)
    ci = d.drum.index("H")
    d.seed_char_error(5, ci, 32)  # group 2, local 2: H shows next char
    _calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.displayed_char(5, "H") == "H"
    assert d.remote_char[0][2][ci] == 24
    assert any(p[0] == 2 and p[1] == "char" for p in d.persists)


def test_stuck_module_escalates(tmp_path):
    d = FakeDisplay(total=4)
    d.seed_module_error(2, -d.spc)
    reader = SimReader(d)
    reader.frozen[2] = "X"  # camera sees this module never move
    calib = Calibrator(d, FakeCamera(), reader, photo_dir=str(tmp_path),
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
    calib = Calibrator(d, FakeCamera(), SimReader(d),
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
    assert any("full shift -1" in e["text"] for e in events)


def test_sweep_covers_every_character_once_in_reverse(tmp_path):
    # P1 measures the whole drum: every character is commanded once on
    # every module, in reverse drum order.
    d = FakeDisplay(total=4, charset=48)
    events = []
    calib = Calibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                       dwell_ms=0, timeout_s=5, min_confidence=0.5,
                       mode="full", on_event=events.append)
    calib.run()
    shown = [e["text"].split(":", 1)[0] for e in events
             if e["kind"] == "read" and e["text"].startswith("sw_")]
    assert len(shown) == len(d.drum)
    assert len(set(shown)) == len(d.drum)  # no character re-shown


def test_sweep_budget_counts_passes_and_fires(tmp_path):
    # One sweep pass = one sweep of the budget; the guard fires.
    d = FakeDisplay(total=4)
    calib = Calibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
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
    d = FakeDisplay(total=4)
    calib = Calibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                       dwell_ms=0, timeout_s=5, min_confidence=0.5,
                       mode="full")
    calib.max_sweeps = 0
    report = calib.run()
    assert report["result"] == "needs-human"
    assert "sweep budget exhausted" in report["reason"]


def test_ladder_background_fillers_rotate(tmp_path):
    # Background slots must move every frame: a constant filler stalls a
    # misaligned module (the firmware resolves the repeat to identical
    # step positions and every re-read scores the same stale flap).
    d = FakeDisplay(total=4, charset=48)
    calib = Calibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                       dwell_ms=0, timeout_s=5, min_confidence=0.5,
                       mode="full")
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    plans = [{"module": 0, "group": 1, "local": 0, "char_index": -1,
              "steps": [4], "absolute": True, "cap": d.spc,
              "targets": ["E"], "guards": ["H"],
              "state": 0, "best": 0, "best_score": 0, "label": "test"}]
    calib._cell_ladder(plans)
    shown = [e["frame"] for e in calib.frames
             if e["tag"].startswith("ladder_")]
    assert len(shown) >= 2
    for frame in shown:
        # Module 3 is background in every ladder frame.
        assert frame[3] in d.drum
    assert any(a[3] != b[3] for a, b in pairwise(shown))


def test_summary_records_offset_delta(tmp_path):
    d = FakeDisplay(total=4)
    d.seed_module_error(1, -d.spc)
    _, report = run_calib(d, tmp_path)
    m1 = next(m for m in report["summary"]["modules"] if m["module"] == 1)
    assert m1["offset_delta"] == -d.spc
    assert not m1["persistent_wrong_glyph"]


def test_read_log_line_reports_expected_vs_read(tmp_path):
    # The read event must show the commanded frame, the transcription and
    # every mismatched module.
    d = FakeDisplay(total=4)
    d.seed_module_error(1, -d.spc)
    events = []
    calib = Calibrator(d, FakeCamera(), SimReader(d),
                       photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                       min_confidence=0.5, mode="full",
                       on_event=events.append)
    calib.run()
    first = next(e for e in events if e["kind"] == "read")
    assert "want" in first["text"] and "saw" in first["text"]
    assert "m1 want" in first["text"]  # the off module is named


def test_fine_identity_that_does_not_fit_escalates(tmp_path):
    # A per-character fault that needs a whole flap is unreachable from
    # any candidate inside the ±32 clamp: probe, find nothing, escalate.
    d = FakeDisplay(total=4)
    ci = d.drum.index("O")
    d.seed_char_error(2, ci, 30 * d.spc)
    _calib, report = run_calib(d, tmp_path)
    assert report["result"] == "needs-human"
    notes = [e["note"] for e in report["identity"]["persistent"]]
    assert any("incremental ladder" in n and "char cell" in n for n in notes)
    assert d.char_off[2].get(ci, 0) == 30 * d.spc  # untouched


def test_confusable_pair_never_becomes_a_correction(tmp_path):
    # O and 0 are indistinguishable on the drum: a reader that swaps them
    # must not be "corrected". Confusable differences are no-information.
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
    calib = Calibrator(d, FakeCamera(), ConfusedReader(d),
                       photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                       min_confidence=0.5, mode="full")
    report = calib.run()
    assert report["result"] == "converged"
    assert d.mod_off == [0] * d.local
    assert not d.persists


def test_cell_ladder_rejects_non_improving_nudge(tmp_path):
    # A candidate that leaves the target just as wrong must NOT persist a
    # no-op offset.
    d = FakeDisplay(total=4, charset=48)
    ci = d.drum.index("E")
    d.seed_char_error(0, ci, 32)  # 'E' one flap ahead; -4 cannot fix it
    calib = Calibrator(d, FakeCamera(), SimReader(d),
                       photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                       min_confidence=0.5, mode="full")
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    plans = [{"module": 0, "group": 1, "local": 0, "char_index": ci,
              "steps": [-4], "absolute": True, "cap": CHAR_OFFSET_LIMIT,
              "targets": ["E"], "guards": [],
              "state": 0, "best": 0, "best_score": 0, "label": "test"}]
    assert calib._cell_ladder(plans) == set()
    assert plans[0]["best"] == 0
    assert not d.persists
    assert d.previews  # the candidate was applied and reverted
    assert d.char_off[0].get(ci, 0) == 32  # untouched


def test_p2_ladder_scans_beyond_the_old_ceiling(tmp_path):
    # P2's incremental ladder scans in steps of 4 up to the firmware's ±32
    # clamp, so a fault beyond the old ladder's ceiling is still found.
    d = FakeDisplay(total=4, charset=48)
    ci = d.drum.index("E")
    d.seed_char_error(0, ci, -32)
    events = []
    calib = Calibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                       dwell_ms=0, timeout_s=5, min_confidence=0.5,
                       mode="full", on_event=events.append)
    report = calib.run()
    assert report["result"] == "converged"
    assert d.displayed_char(0, "E") == "E"
    assert d.char_off[0].get(ci, 0) == -20  # base -32 + the +12 candidate
    assert any("cell ladder" in e["text"] and "+12 steps" in e["text"]
               for e in events)


def test_p2_ladder_stops_probing_a_fixed_cell(tmp_path):
    # A candidate that makes the target read correctly is final: the cell
    # must stop being nudged instead of burning the preview budget.
    d = FakeDisplay(total=4, charset=48)
    ci = d.drum.index("E")
    d.seed_char_error(0, ci, -24)  # previous glyph; +4 lands it back
    calib, report = run_calib(d, tmp_path)
    assert report["result"] == "converged"
    assert d.char_off[0].get(ci, 0) == -20  # base -24 + the +4 candidate
    assert d.previews == [(0, ci, 4)]
    ladder = [e for e in calib.frames if e["tag"].startswith("ladder_")]
    assert len(ladder) == 2  # baseline + the one candidate round
    assert d.persists == [(1, "char", -20, 0, ci)]


def test_p2_ladder_steps_default_and_explicit_cap():
    # The seam default caps at half a character pitch; identity faults
    # pass the firmware's ±32 char-cell clamp so the scan can reach it.
    steps = _p2_ladder_steps(43)
    assert max(abs(s) for s in steps) == 20  # half-pitch cap (43 // 2 = 21)
    assert steps[-4:] == [2, -2, 1, -1]      # narrow-window fallback
    wide = _p2_ladder_steps(43, cap=32)
    assert wide[:4] == [4, -4, 8, -8]
    assert max(abs(s) for s in wide) == 32
    assert wide[-4:] == [2, -2, 1, -1]


def test_p2_ladder_respects_char_cell_clamp(tmp_path):
    # Char-cell candidates are bounded by the firmware's ±32 clamp: the
    # ladder never probes a landing outside the window.
    d = FakeDisplay(total=4, charset=48)
    ci = d.drum.index("E")
    d.seed_char_error(0, ci, -30)
    calib = Calibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                       dwell_ms=0, timeout_s=5, min_confidence=0.5,
                       mode="full")
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    calib._p1_flagged = ["E"]
    calib._p2_fine()
    assert d.displayed_char(0, "E") == "E"
    assert d.char_off[0].get(ci, 0) == -18  # base -30 + the +12 candidate
    value = -30
    for _, cell, delta in d.previews:
        if cell < 0:
            continue
        value += delta
        assert -32 <= value <= 32


def test_ladder_stops_before_the_budget_wall_and_commits(tmp_path):
    # A ladder can run out of preview budget with candidates still
    # untested: it must stop probing while a round is still affordable and
    # let the commit loop persist the offsets already verified.
    d = FakeDisplay(total=4, charset=48)
    ci = d.drum.index("E")
    d.seed_char_error(0, ci, -24)  # +4 reads it correctly: settles round 0
    d.seed_char_error(1, ci, 26)   # still wrong after every +\-4 candidate
    events = []
    calib = Calibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                       dwell_ms=0, timeout_s=5, min_confidence=0.5,
                       mode="full", on_event=events.append)
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    # Round 0 costs three previews (both applies, then the loser's
    # revert); a second round needs more than the four allowed.
    calib.max_previews = 4
    steps = _p2_ladder_steps(d.spc, cap=CHAR_OFFSET_LIMIT)
    plans = [
        {"module": 0, "group": 1, "local": 0, "char_index": ci,
         "steps": steps, "absolute": True, "cap": CHAR_OFFSET_LIMIT,
         "targets": ["E"], "guards": [],
         "state": 0, "best": 0, "best_score": 0, "label": "winner"},
        {"module": 1, "group": 1, "local": 1, "char_index": ci,
         "steps": steps, "absolute": True, "cap": CHAR_OFFSET_LIMIT,
         "targets": ["E"], "guards": [],
         "state": 0, "best": 0, "best_score": 0, "label": "stuck"},
    ]
    assert calib._cell_ladder(plans) == {0}
    assert d.char_off[0].get(ci, 0) == -20  # the verified winner, committed
    assert d.char_off[1].get(ci, 0) == 26   # nothing invented for the stuck
    assert any("stopped before step" in e["text"] for e in events)
    assert len(d.previews) <= calib.max_previews


def test_ladder_filler_rotates_background_slots(tmp_path):
    # A constant filler stalls a misaligned background module (identical
    # step positions -> zero commanded steps -> stale flap re-read).
    # Background slots must change glyph on every consecutive ladder frame.
    d = FakeDisplay(total=4, charset=48)
    calib = Calibrator(d, FakeCamera(), SimReader(d),
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
    for prev, cur in pairwise(shown):
        # Module 3 is background in every frame: must always move.
        assert prev[3] != cur[3], f"background stalled: {prev!r} -> {cur!r}"


def test_ladder_skips_reshow_when_nothing_moved(tmp_path):
    # A plan cell whose state did not change keeps its cached score: no
    # new frames are burned re-reading it.
    d = FakeDisplay(total=4, charset=48)
    calib = Calibrator(d, FakeCamera(), SimReader(d),
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


def test_proportional_residual_applies_fractional_module_fix(tmp_path):
    # +1 on 32 of 48 characters (66.7%) falls in the proportional band:
    # round(2048/48 * 32/48 / 0.75) = +38 steps, not a whole character. (14 read correct via cancelling char errors; two sweep
    # samples are systematically excluded: a '?' read trips the
    # unknown-sentinel trust rule and '%' wraps to blank.) The 5-step
    # remainder is sub-pitch and reads clean.
    d = FakeDisplay(total=4, charset=48)
    d.seed_module_error(0, d.spc)  # whole drum one character ahead
    minority = list(d.drum[1:15])
    for ch in minority:
        d.seed_char_error(0, d.drum.index(ch), -d.spc)
    events = []
    calib = Calibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                       dwell_ms=0, timeout_s=5, min_confidence=0.5,
                       mode="full", on_event=events.append)
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    calib._p1_coarse()
    assert d.mod_off[0] == -5, d.mod_off
    assert not any("unreliable reads" in e["text"] for e in events)
    assert any("proportional shift +1" in e["text"] for e in events)


def test_two_sided_residuals_apply_dominant_proportion(tmp_path):
    # Gaps in both directions: the dominant one (+1 on 14/48 = 29.2%)
    # earns round(2048/48 * 14/48 / 0.75) = +17 steps while the minority
    # direction becomes P2 work. Neither direction escalates.
    d = FakeDisplay(total=4, charset=48)
    for ch in list(d.drum[1:15]):
        d.seed_char_error(0, d.drum.index(ch), d.spc)
    for ch in list(d.drum[15:24]):
        d.seed_char_error(0, d.drum.index(ch), -d.spc)
    events = []
    calib = Calibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                       dwell_ms=0, timeout_s=5, min_confidence=0.5,
                       mode="full", on_event=events.append)
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    calib._p1_coarse()
    assert d.mod_off[0] == 17, d.mod_off
    assert not any("unreliable reads" in e["text"] for e in events)
    assert any("proportional shift +1" in e["text"] for e in events)


def test_blank_read_against_nonblank_command_is_junk(tmp_path):
    # A blank read against a non-blank command is a failed sampling frame
    # -> excluded (None), not a residual.
    d = FakeDisplay(total=4, charset=48)
    calib = Calibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                       dwell_ms=0, timeout_s=5, min_confidence=0.5,
                       mode="full")
    calib.total, calib.drum = d.total, d.drum
    blank_against_colon = ModuleReading(0, " ", "blank", 0.9, "vlm", ":")
    assert calib._shift(blank_against_colon, ":") is None
    blank_against_blank = ModuleReading(0, " ", "clean", 0.9, "vlm", " ")
    assert calib._shift(blank_against_blank, " ") == 0


def test_single_flap_arc_gets_proportional_module_fix(tmp_path):
    # A same-sign single-flap arc (13/48 = 27%) earns a proportional
    # module correction — round(2048/48 * 13/48 / 0.75) = +15 steps —
    # and is NOT escalated as reader noise. Whatever still reads wrong
    # after the move is ordinary per-character P2 work.
    d = FakeDisplay(total=4, charset=48)
    arc = list(d.drum[1:14])  # 13 chars = 27% of the drum
    for ch in arc:
        d.seed_char_error(0, d.drum.index(ch), d.spc)  # one flap ahead
    events = []
    calib = Calibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                       dwell_ms=0, timeout_s=5, min_confidence=0.5,
                       mode="full", on_event=events.append)
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    calib._p1_coarse()
    assert d.mod_off[0] == 15, d.mod_off
    assert d.persists
    assert not any("unreliable reads" in e["text"] for e in events)
    assert any("proportional shift +1" in e["text"] for e in events)
    assert set(arc) <= set(calib._p1_flagged)  # leftovers are P2 work


def test_p1_direction_prefers_larger_share_and_minimum_gap():
    # Same-direction magnitudes combine; the minimum gap is the module's
    # magnitude; ties go positive; a clean histogram has no direction.
    assert _p1_direction(Counter({1: 20, 2: 2})) == (1, 22, 1)
    assert _p1_direction(Counter({-1: 10, 1: 15})) == (1, 15, 1)
    assert _p1_direction(Counter({-2: 3, -1: 4})) == (-1, 7, 1)
    assert _p1_direction(Counter({1: 5, -1: 5})) == (1, 5, 1)
    assert _p1_direction(Counter({0: 48})) == (0, 0, 0)
    assert _p1_direction(Counter()) == (0, 0, 0)


def test_p1_module_delta_bands():
    pitch = 2048 / 48
    assert _p1_module_delta(1.0, 1, pitch) == 43      # full band
    assert _p1_module_delta(0.75, 1, pitch) == 43     # boundary inclusive
    assert _p1_module_delta(22 / 48, 1, pitch) == 26  # 20x+1 + 2x+2 example
    assert _p1_module_delta(0.5, 1, pitch) == 28
    assert _p1_module_delta(0.125, 1, pitch) == 7     # boundary inclusive
    assert _p1_module_delta(0.124, 1, pitch) == 0     # below -> P2 work
    assert _p1_module_delta(0.0, 1, pitch) == 0
    assert _p1_module_delta(1.0, 0, pitch) == 0
    assert _p1_module_delta(1.0, 2, pitch) == 85      # n=2 full multiples
    assert _p1_module_delta(1.0, -1, pitch) == -43    # sign preserved
    assert _p1_module_delta(0.5, -1, pitch) == -28


def test_p1_combines_same_direction_magnitudes_using_minimum_gap(tmp_path):
    # 20x +1 plus 2x +2 count together (22/48 = 45.8%) with n = 1:
    # round(2048/48 * 22/48 / 0.75) = +26 steps on the module cell.
    d = FakeDisplay(total=4, charset=48)
    plus_one = list(d.drum[1:21])
    plus_two = list(d.drum[21:23])
    for ch in plus_one:
        d.seed_char_error(0, d.drum.index(ch), d.spc)
    for ch in plus_two:
        d.seed_char_error(0, d.drum.index(ch), 2 * d.spc)
    calib = Calibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                       dwell_ms=0, timeout_s=5, min_confidence=0.5,
                       mode="full")
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    calib._p1_coarse()
    mod_previews = [(ci, delta) for _, ci, delta in d.previews if ci < 0]
    assert mod_previews == [(-1, 26)]
    assert d.mod_off[0] == 26
    # The +1 bulk reads clean after the move; the +2 tail is P2 work.
    assert set(plus_two) <= set(calib._p1_flagged)
    assert not (set(plus_one) & set(calib._p1_flagged))


def test_report_written_to_disk(tmp_path):
    d = FakeDisplay(total=4)
    run_calib(d, tmp_path)
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "snapshot.json").is_file()


def test_aborts_after_too_many_unreliable_reads(tmp_path):
    # More than MAX_UNRELIABLE_READS reader-trust escalations must stop the
    # run immediately instead of tuning noise.
    d = FakeDisplay(total=4)
    events = []
    calib = Calibrator(d, FakeCamera(), SimReader(d),
                       photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                       min_confidence=0.5, mode="full",
                       on_event=events.append)
    for i in range(MAX_UNRELIABLE_READS):
        calib._escalate(0, "?", f"unreliable reads ({i} samples)")
    assert calib._unreliable_count == MAX_UNRELIABLE_READS
    with pytest.raises(CalibError, match="too many unreliable reads"):
        calib._escalate(0, "?", "unreadable during fine pass")


def test_unreadable_escalation_counts_toward_abort(tmp_path):
    d = FakeDisplay(total=4)
    calib = Calibrator(d, FakeCamera(), SimReader(d),
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
    # event; status polls stay unlogged.
    d = FakeDisplay(total=4)
    events = []
    calib = Calibrator(d, FakeCamera(), SimReader(d),
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
    calib = Calibrator(d, FakeCamera(), reader,
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
    # carry monotonic seconds since run start.
    d = FakeDisplay(total=4)
    events = []
    calib = Calibrator(d, FakeCamera(), SimReader(d),
                       photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                       min_confidence=0.5, mode="full",
                       on_event=events.append)
    calib.event("phase", "probe phase")
    assert events[0]["elapsed"] >= 0.0
    assert isinstance(events[0]["elapsed"], float)


def test_report_has_timing_context_and_traceback(tmp_path):
    # Root-cause + speed analysis from report.json alone: run config,
    # firmware identity, wall seconds, per-phase costs.
    d = FakeDisplay(total=4)
    d.seed_module_error(1, -d.spc)
    _calib, report = run_calib(d, tmp_path)
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
                          "previews", "persists"}
               for p in report["timing"]["phases"])
    # Each row carries the cost of the phase NAMED in it: the last row is
    # the last phase that ran, not the trailing cleanup mark.
    assert "acceptance" in phases[-1]


def test_phase_timing_labels_each_row_with_its_own_phase(tmp_path):
    # Marks announce the phase they START, so a slice belongs to the
    # earlier mark: a phase's row must carry that phase's own cost.
    d = FakeDisplay(total=4)
    calib = Calibrator(d, FakeCamera(), SimReader(d),
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
    with pytest.raises(ValueError, match="unknown phases"):
        Calibrator(d, FakeCamera(), SimReader(d),
                   photo_dir=str(tmp_path), phases=["p1", "nope"])


def test_preview_mode_is_rejected_not_supported(tmp_path):
    # The calibrator accepts dry-run/full only.
    d = FakeDisplay(total=4)
    with pytest.raises(ValueError, match="mode must be dry-run or full"):
        Calibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                   dwell_ms=0, timeout_s=5, min_confidence=0.5,
                   mode="preview")


def test_p1_only_skips_later_phases(tmp_path):
    # P1-only: module offsets commit, P2/P4/acceptance never run, and the
    # report marks them skipped with a subset verdict.
    d = FakeDisplay(total=4)
    d.seed_module_error(1, -d.spc)
    _calib, report = run_calib(d, tmp_path, phases=["p1"])
    assert report["phases"] == ["p1"]
    assert report["skipped"] == ["p2", "p4", "acceptance"]
    assert d.mod_off[1] == 0
    assert d.persists
    assert report["result"] == "needs-human"
    assert "acceptance skipped" in report["reason"]


def test_p2_without_p1_derives_flagged_readonly(tmp_path):
    # P2 without P1: a read-only sweep derives the flagged chars (no
    # module commits), then P2 tunes the per-char cell.
    d = FakeDisplay(total=4)
    ci = d.drum.index("O")
    d.seed_char_error(2, ci, -d.spc)  # only 'O' is one char behind on m2
    _calib, report = run_calib(d, tmp_path, phases=["p2"])
    assert report["phases"] == ["p2"]
    assert report["skipped"] == ["p1", "p4", "acceptance"]
    assert d.displayed_char(2, "O") == "O"
    assert abs(d.char_off[2].get(ci, 0)) <= 32
    # No module-cell writes: the read-only sweep must not commit.
    assert all(x["charIndex"] >= 0 for x in report["deltas"])
    assert d.mod_off == [0] * d.local


def test_p4_only_is_readonly(tmp_path):
    # P4/acceptance-only: no previews or persists, verdict escalates.
    d = FakeDisplay(total=4)
    _calib, report = run_calib(d, tmp_path, phases=["p4"])
    assert report["phases"] == ["p4"]
    assert report["skipped"] == ["p1", "p2", "acceptance"]
    assert d.previews == []
    assert d.persists == []
    assert report["result"] == "needs-human"


def test_transient_fine_frame_failure_is_retried(tmp_path):
    # A hard reader failure on a P2 frame must not escalate every module
    # of that frame: the frame is re-read once.
    class FlakyReader(SimReader):
        def __init__(self, display):
            super().__init__(display)
            self.failed = False

        def read(self, jpeg, total, expected="", charset="", drum=""):
            if not self.failed and expected and set(expected) == {"C"}:
                self.failed = True
                raise ReaderError("transient provider failure")
            return super().read(jpeg, total, expected, charset, drum)

    d = FakeDisplay(total=4, charset=48)
    ci = d.drum.index("C")
    d.seed_char_error(0, ci, 32)  # 'C' shows the next flap
    calib = Calibrator(d, FakeCamera(), FlakyReader(d), str(tmp_path),
                       dwell_ms=0, timeout_s=5, min_confidence=0.5,
                       mode="full")
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    calib._p1_flagged = ["C"]
    calib._p2_fine()
    assert calib.identity_persistent == []   # no bogus escalations
    assert d.displayed_char(0, "C") == "C"   # retry read it, ladder fixed
    assert d.char_off[0].get(ci, 0) == 20    # base 32 + accepted -12


def test_persistent_fine_frame_failure_escalates_once_per_module(tmp_path):
    # A frame the reader never answers (the retry fails too) is one reader
    # failure per module, not one per (module, character) pair.
    class DeadReader(SimReader):
        def read(self, jpeg, total, expected="", charset="", drum=""):
            if expected and set(expected) == {"C"}:
                raise ReaderError("provider down")
            return super().read(jpeg, total, expected, charset, drum)

    d = FakeDisplay(total=4, charset=48)
    ci = d.drum.index("C")
    d.seed_char_error(0, ci, 32)
    calib = Calibrator(d, FakeCamera(), DeadReader(d), str(tmp_path),
                       dwell_ms=0, timeout_s=5, min_confidence=0.5,
                       mode="full")
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    calib._p1_flagged = ["C", "D"]
    calib._p2_fine()
    fine = [e for e in calib.identity_persistent if "fine pass" in e["note"]]
    assert [e["module"] for e in fine] == [0, 1, 2, 3]  # once per module
    assert calib._unreliable_count == 4  # below MAX: no abort


def test_unreliable_reads_abort_the_run(tmp_path):
    # A reader that reports its own geometry failed (module grid not on the
    # modules) has nothing worth tuning on: three frames in a row abort with
    # the reader's note, instead of a whole sweep producing plausible-looking
    # garbage that escalates module by module.
    d = FakeDisplay(total=4)
    calib = Calibrator(d, FakeCamera(), SimReader(d, unreliable=True),
                       str(tmp_path), dwell_ms=0, timeout_s=5,
                       min_confidence=0.5, mode="full")
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    with pytest.raises(CalibError) as excinfo:
        calib._p0_register()
    assert "reader unreliable" in str(excinfo.value)
    assert calib._reader_failures == 3  # p0_blank, p0_allH, p0_index


def test_reader_failure_count_tracks_trust(tmp_path):
    # Only a trustworthy read clears the failure count: an unreliable read
    # increments it, so a reader that alternates between good and bad still
    # aborts instead of running forever.
    d = FakeDisplay(total=4)
    reader = SimReader(d, unreliable=True)
    calib = Calibrator(d, FakeCamera(), reader, str(tmp_path), dwell_ms=0,
                       timeout_s=5, min_confidence=0.5, mode="full")
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    calib._reader_failures = 2
    with pytest.raises(CalibError):
        calib._show_read(" " * d.total, "probe")  # 2 -> 3 abort
    reader.unreliable = False
    calib._reader_failures = 1
    calib._show_read(" " * d.total, "probe2")
    assert calib._reader_failures == 0  # a good read clears it


def test_budget_stopped_ladder_reports_truncation_not_hardware(tmp_path):
    # A ladder that stops on budget before probing a cell must not be
    # reported as "no working offset ... tried 0 candidate(s)".
    d = FakeDisplay(total=4, charset=48)
    ci = d.drum.index("C")
    d.seed_char_error(0, ci, 32)
    events = []
    calib = Calibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                       dwell_ms=0, timeout_s=5, min_confidence=0.5,
                       mode="full", on_event=events.append)
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    calib.max_previews = 1  # cannot afford even one apply/revert round
    calib._p1_flagged = ["C"]
    calib._p2_fine()
    assert not any("no working offset" in e["note"]
                   for e in calib.identity_persistent)
    assert any("stopped on budget" in e["text"] for e in events)
    assert d.char_off[0].get(ci, 0) == 32  # untouched


def test_cell_ladder_rejects_two_plans_on_one_module(tmp_path):
    # Scores and frame slots are keyed by module, so two plans on one
    # module would share a score and could both commit off one verified
    # reading. The ladder must refuse.
    d = FakeDisplay(total=4, charset=48)
    calib = Calibrator(d, FakeCamera(), SimReader(d), str(tmp_path),
                       dwell_ms=0, timeout_s=5, min_confidence=0.5,
                       mode="full")
    calib.total, calib.drum = d.total, d.drum
    calib.steps_per_char = d.spc
    calib.group_widths = [4]
    plan = {"module": 0, "group": 1, "local": 0, "char_index": -1,
            "steps": [4], "absolute": True, "cap": d.spc,
            "targets": ["E"], "guards": [], "state": 0, "best": 0,
            "best_score": 0, "label": "a"}
    with pytest.raises(ValueError, match="one plan per module"):
        calib._cell_ladder([dict(plan), dict(plan, label="b")])
