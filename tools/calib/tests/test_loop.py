"""Tests for calib/loop.py against fakes (no display, no camera)."""

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


def test_phase2_dry_run_records_proposals_only(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "split_crops",
                        lambda gray, n: [gray[:, i * 68:(i + 1) * 68] for i in range(n)])
    cal = _calibrator(2, tmp_path)
    report = cal.run()
    assert report["result"] in ("converged", "needs-human")
    assert cal.display.persists == []  # dry run: NVS untouched


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
