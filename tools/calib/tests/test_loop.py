"""Tests for calib/loop.py against fakes (no display, no camera)."""

import json

import numpy as np

from calib import vision
from calib.loop import Calibrator


class FakeDisplay:
    """All-ok display: every show settles instantly, offsets start at 0."""

    def __init__(self, total=4, charset=37):
        self.total = total
        self.charset = charset
        self.drum = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"[:charset]
        self.frame_ids = 0
        self.previews = []
        self.persists = []
        self.live = {}  # (scope, module, char) -> value

    def status(self):
        return {"totalModules": self.total, "numModules": self.total,
                "groupCount": 1, "charset": self.charset,
                "drumOrder": self.drum, "contractVersion": 1,
                "moduleOffsets": [0] * self.total}

    def contract(self):
        return {"contractVersion": 1}

    def hold(self, active):
        return {"holdActive": active}

    def snapshot(self):
        return {"settings": {"mode": 0}}

    def restore(self, snapshot):
        return {"message": "restored"}

    def show_and_settle(self, frame, dwell_ms=800, timeout_s=60):
        self.frame_ids += 1
        return {"frameId": self.frame_ids, "fleetFrame": False}

    def wait_settled(self, timeout_s=60):
        return self.status()

    def frame_info(self, frame_id):
        return {"settled": True}

    def preview(self, module, char_index, delta):
        self.previews.append((module, char_index, delta))
        return {"message": "queued"}

    def persist(self, scope, kind, value, module=0, char_index=0):
        self.persists.append((scope, kind, value, module, char_index))
        self.live[(scope, module, char_index)] = value
        return {"message": "saved"}


class FakeCamera:
    """Returns a frame with per-module brightness variation (P0-distinct)
    and no seam gradients, so every crop scores ok."""

    def __init__(self, total=4):
        self.total = total

    def capture(self):
        h, w, n = 96, 68, self.total
        frame = np.zeros((h, w * n), dtype=np.uint8)
        for i in range(n):
            frame[:, i * w : (i + 1) * w] = 150 + i * 10
        return frame


def _calibrator(phase, tmp_path):
    return Calibrator(FakeDisplay(), FakeCamera(), photo_dir=str(tmp_path),
                      dwell_ms=0, timeout_s=5, max_phase=phase)


def test_phase1_readonly_converges_without_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "split_crops",
                        lambda gray, n: [gray[:, i * 68:(i + 1) * 68] for i in range(n)])
    cal = _calibrator(1, tmp_path)
    report = cal.run()
    assert report["result"] == "converged", report.get("reason")
    assert cal.display.previews == []
    assert cal.display.persists == []
    assert (tmp_path / "report.json").exists()
    # Summary block: machine-checkable per-module outcome.
    s = report["summary"]
    assert s["ok"] is True and s["result"] == "converged"
    assert s["persistent_wrong_glyph_modules"] == []
    assert [m["module"] for m in s["modules"]] == [0, 1, 2, 3]
    assert all(m["offset_delta"] == 0 for m in s["modules"])  # read-only run


def test_phase2_dry_run_records_proposals_only(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "split_crops",
                        lambda gray, n: [gray[:, i * 68:(i + 1) * 68] for i in range(n)])
    cal = _calibrator(2, tmp_path)
    report = cal.run()
    assert report["result"] in ("converged", "needs-human")
    assert cal.display.persists == []  # dry run: NVS untouched


def test_widths_inconsistent_geometry_raises(tmp_path):
    # Firmware groupCount bug (integer division 12/8=1) used to map
    # remote modules to out-of-range local indices and die mid-run with
    # a cryptic preview 400. The geometry check must name the problem.
    import pytest

    from calib.display import CalibError

    display = FakeDisplay(total=12)
    display.status = lambda: {  # groupCount lies: 1 group for 12 modules
        "totalModules": 12, "numModules": 8, "groupCount": 1,
        "charset": 37, "drumOrder": display.drum, "contractVersion": 1,
        "moduleOffsets": [0] * 8}
    cal = Calibrator(display, FakeCamera(total=12), photo_dir=str(tmp_path),
                     dwell_ms=0, timeout_s=5, max_phase=1)
    with pytest.raises(CalibError, match="geometry inconsistent"):
        cal.run()


def test_widths_fleet_split_maps_remote_modules(tmp_path):
    # Correct firmware geometry (groupCount 2: 8 local + 4 remote):
    # global module 8 must map to group 2, local index 0.
    display = FakeDisplay(total=12)
    display.status = lambda: {
        "totalModules": 12, "numModules": 8, "groupCount": 2,
        "charset": 37, "drumOrder": display.drum, "contractVersion": 1,
        "moduleOffsets": [0] * 8}
    cal = Calibrator(display, FakeCamera(total=12), photo_dir=str(tmp_path),
                     dwell_ms=0, timeout_s=5, max_phase=1)
    cal.total = 12
    cal.charset = 37
    cal.drum = display.drum
    cal.group_widths = cal._widths(display.status())
    assert cal.group_widths == [8, 4]
    assert cal._group_of(7) == 1
    assert cal._group_of(8) == 2
    assert cal._local_index(8) == 0
    assert cal._local_index(11) == 3
    # Out-of-fleet indices raise instead of silently mapping to group 1.
    import pytest

    from calib.display import CalibError

    with pytest.raises(CalibError, match="outside fleet geometry"):
        cal._group_of(12)


class WrongGlyphCamera(FakeCamera):
    """Module 3 shows a different (but clean) glyph on every frame."""

    def capture(self):
        from tests.fixtures import make_glyph

        return np.concatenate(
            [make_glyph("E"), make_glyph("E"), make_glyph("E"), make_glyph("I")], axis=1)


def test_wrong_glyph_escalates_to_needs_human(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "split_crops",
                        lambda gray, n: [gray[:, i * 64:(i + 1) * 64] for i in range(n)])
    display = FakeDisplay()
    cal = Calibrator(display, WrongGlyphCamera(), photo_dir=str(tmp_path),
                     dwell_ms=0, timeout_s=5, max_phase=1)
    report = cal.run()
    assert report["result"] == "needs-human"
    assert any(e["module"] == 3 for e in report["identity"]["persistent"])
    assert "wrong glyph" in report["reason"]
    # Summary flags the defective module and fails the machine check.
    s = report["summary"]
    assert s["ok"] is False
    assert s["persistent_wrong_glyph_modules"] == [3]
    assert s["modules"][3]["persistent_wrong_glyph"] is True
    assert not s["modules"][0]["persistent_wrong_glyph"]


class SystematicShiftCamera(FakeCamera):
    """Every module shows glyph I no matter what is commanded (systematic
    all-agree-but-wrong shift). Per-module brightness offsets keep P0
    registration passing; ZNCC mean-normalization keeps consensus blind."""

    def capture(self):
        from tests.fixtures import make_glyph

        parts = []
        for i in range(4):
            crop = make_glyph("I").astype(int) + i * 8
            parts.append(np.clip(crop, 0, 255).astype("uint8"))
        return np.concatenate(parts, axis=1)


def test_systematic_shift_caught_by_golden_bank(tmp_path, monkeypatch):
    from tests.fixtures import make_glyph

    monkeypatch.setattr(vision, "split_crops",
                        lambda gray, n: [gray[:, i * 64:(i + 1) * 64] for i in range(n)])
    # Golden bank with ONLY the E template: top hit is E by default, so only
    # the absolute score floor can catch the all-I display.
    bank = vision.build_template_bank({"E": [make_glyph("E")] * 3})
    vision.save_template_bank(bank, {"source": "golden-test"}, str(tmp_path / "templates"))
    cal = Calibrator(FakeDisplay(), SystematicShiftCamera(), photo_dir=str(tmp_path),
                     dwell_ms=0, timeout_s=5, max_phase=1)
    report = cal.run()
    assert report["result"] == "needs-human"
    assert report["identity"]["bank"]["source"] == "golden-test"
    assert "wrong glyph" in report["reason"]


def test_p2_two_staggered_passes_cover_two_residue_classes(tmp_path, monkeypatch):
    # Every drum character must appear, and each module must be exercised
    # on both residue classes (offsets 0 and stride//2) — not just one.
    monkeypatch.setattr(vision, "split_crops",
                        lambda gray, n: [gray[:, i * 68:(i + 1) * 68] for i in range(n)])
    cal = _calibrator(2, tmp_path)
    cal.run()
    drum = cal.drum
    n = len(drum)
    p2 = [f for f in cal.frames if f["tag"].startswith("p2_stride")]
    assert len(p2) == len(range(0, n, 6)) + len(range(3, n, 6))
    for i in range(cal.total):
        residues = {(drum.index(f["frame"][i]) - i) % 6 for f in p2}
        # Superset (drum length need not divide the stride, so wraparound
        # frames may leak extra residues); both passes must be present.
        assert {0, 3} <= residues, f"module {i}: residues {sorted(residues)}"


def test_p2_full_mode_covers_every_residue_class(tmp_path, monkeypatch):
    # --full: all six stride offsets, so each drum character lands on each
    # module; budgets must scale (no sweep-budget abort mid-run).
    monkeypatch.setattr(vision, "split_crops",
                        lambda gray, n: [gray[:, i * 68:(i + 1) * 68] for i in range(n)])
    cal = Calibrator(FakeDisplay(), FakeCamera(), photo_dir=str(tmp_path),
                     dwell_ms=0, timeout_s=5, max_phase=2, full=True)
    report = cal.run()
    assert "budget exhausted" not in report.get("reason", "")
    assert report["full"] is True
    drum = cal.drum
    n = len(drum)
    p2 = [f for f in cal.frames if f["tag"].startswith("p2_stride")]
    assert len(p2) == sum(len(range(o, n, 6)) for o in range(6))
    for i in range(cal.total):
        residues = {(drum.index(f["frame"][i]) - i) % 6 for f in p2}
        assert set(range(6)) <= residues, f"module {i}: {sorted(residues)}"


def test_cli_verify_exit_codes(tmp_path, capsys):
    from calib.cli import main

    good = tmp_path / "run-good"
    good.mkdir()
    (good / "report.json").write_text(json.dumps({
        "result": "converged",
        "summary": {"ok": True, "result": "converged",
                    "persistent_wrong_glyph_modules": [],
                    "modules": [{"module": 0, "offset_delta": 2,
                                 "persistent_wrong_glyph": False,
                                 "acceptance": {"accept_E": "ok"}}]}}))
    assert main(["--verify", str(good)]) == 0
    out = capsys.readouterr().out
    assert "VERIFY OK" in out and "m0: offset_delta=+2" in out

    bad = tmp_path / "run-bad"
    bad.mkdir()
    (bad / "report.json").write_text(json.dumps({
        "result": "needs-human",
        "summary": {"ok": False, "result": "needs-human",
                    "persistent_wrong_glyph_modules": [3],
                    "modules": [{"module": 3, "offset_delta": 0,
                                 "persistent_wrong_glyph": True,
                                 "acceptance": {}}]}}))
    assert main(["--verify", str(bad)]) == 1
    assert "VERIFY FAILED" in capsys.readouterr().err

    # Older reports without a summary: derive persistent from identity.
    old = tmp_path / "run-old"
    old.mkdir()
    (old / "agent_report.json").write_text(json.dumps({
        "result": "converged",
        "identity": {"persistent": [{"module": 1, "glyph": "O"}]}}))
    assert main(["--verify", str(old)]) == 1

    # No report at all: usage error, not a crash.
    empty = tmp_path / "run-empty"
    empty.mkdir()
    assert main(["--verify", str(empty)]) == 2
