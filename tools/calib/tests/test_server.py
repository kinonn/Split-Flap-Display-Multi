"""Tests for calib/server.py (no display, no camera, no network).

Exercises the real UICalibrator/Harness against fakes: event emission,
abort, a full phase-1 run, config round-trip, and the photo path guard.
"""

import json
import os
import threading

import numpy as np
import pytest

from calib.display import CalibError
from calib.server import Harness, UICalibrator, load_config, photo, save_config


class FakeDisplay:
    """All-ok display: every show settles instantly, offsets start at 0."""

    def __init__(self, total=4, charset=37):
        self.total = total
        self.charset = charset
        self.drum = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"[:charset]
        self.frame_ids = 0

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

    def show_and_settle(self, frame, dwell_ms=0, timeout_s=5):
        self.frame_ids += 1
        return {"frameId": self.frame_ids, "fleetFrame": False}

    def wait_settled(self, timeout_s=5):
        return self.status()

    def preview(self, module, char_index, delta):
        return {"message": "queued"}

    def persist(self, scope, kind, value, module=0, char_index=0):
        return {"message": "saved"}


class FakeCamera:
    """Flat-brightness modules (P0-distinct), no seams → every crop ok."""

    def __init__(self, total=4):
        self.total = total

    def capture(self):
        h, w, n = 96, 68, self.total
        frame = np.zeros((h, w * n), dtype=np.uint8)
        for i in range(n):
            frame[:, i * w:(i + 1) * w] = 150 + i * 10
        return frame


def _events():
    got = []
    return got, got.append


def test_shoot_emits_photo_event_with_scores(tmp_path):
    got, sink = _events()
    cal = UICalibrator(FakeDisplay(), FakeCamera(), photo_dir=str(tmp_path),
                       dwell_ms=0, timeout_s=5, max_phase=1, on_event=sink)
    cal.total = 4
    rec = cal.shoot("HHHH", "p0_exposure")
    assert os.path.isfile(rec["photo"])
    assert len(got) == 1
    evt = got[0]
    assert evt["kind"] == "photo"
    assert evt["photo"] == os.path.basename(rec["photo"])
    assert evt["detail"]["frame"] == "HHHH"
    assert [s["module"] for s in evt["detail"]["scores"]] == [0, 1, 2, 3]
    assert all(s["verdict"] == "ok" for s in evt["detail"]["scores"])


def test_abort_flag_raises_before_touching_hardware(tmp_path):
    display = FakeDisplay()
    cal = UICalibrator(display, FakeCamera(), photo_dir=str(tmp_path),
                       dwell_ms=0, abort_flag=lambda: True)
    with pytest.raises(CalibError, match="aborted"):
        cal.shoot("HHHH", "x")
    assert display.frame_ids == 0


def _cfg(**over):
    cfg = {"display_host": "fake", "camera_index": 0, "phase": 1,
           "dwell_ms": 0, "timeout_s": 5, "identity_thresh": 0.85,
           "relearn_templates": True}
    cfg.update(over)
    return cfg


def test_harness_phase1_run_converges(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_DATA", str(tmp_path))
    h = Harness()
    out = h.start(_cfg(), display=FakeDisplay(), camera=FakeCamera())
    assert out["status"] == "running"
    h.thread.join(timeout=120)
    assert not h.thread.is_alive()
    st = h.state()
    assert st["status"] == "done", st["report"]
    assert st["report"]["result"] == "converged", st["report"].get("reason")
    assert os.path.isfile(os.path.join(h.run_dir, "snapshot.json"))
    assert os.path.isfile(os.path.join(h.run_dir, "report.json"))
    assert len(st["photos"]) > 0
    kinds = {e["kind"] for e in st["events"]}
    assert {"run", "phase", "photo"} <= kinds
    assert st["persists"] == 0  # phase 1 is read-only


def test_harness_unreachable_display_reports_needs_human(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_DATA", str(tmp_path))

    class DeadDisplay(FakeDisplay):
        def status(self):
            raise CalibError("GET /api/calib/status failed: refused")

    h = Harness()
    h.start(_cfg(), display=DeadDisplay(), camera=FakeCamera())
    h.thread.join(timeout=60)
    st = h.state()
    assert st["status"] == "failed"
    assert st["report"]["result"] == "needs-human"


def test_double_start_conflicts(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_DATA", str(tmp_path))
    gate = threading.Event()

    class SlowDisplay(FakeDisplay):
        def show_and_settle(self, frame, dwell_ms=0, timeout_s=5):
            assert gate.wait(timeout=60)
            return super().show_and_settle(frame, dwell_ms, timeout_s)

    h = Harness()
    h.start(_cfg(), display=SlowDisplay(), camera=FakeCamera())
    try:
        with pytest.raises(Exception, match="already in progress"):
            h.start(_cfg(), display=FakeDisplay(), camera=FakeCamera())
    finally:
        gate.set()
        h.thread.join(timeout=120)


def test_photo_rejects_path_traversal():
    with pytest.raises(Exception, match="bad photo name"):
        photo("../report.json")
    with pytest.raises(Exception, match="bad photo name"):
        photo(".hidden")


def test_config_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_DATA", str(tmp_path))
    save_config({"display_host": "splitflap.lan", "phase": 2, "dwell_ms": 500})
    cfg = load_config()
    assert cfg["display_host"] == "splitflap.lan"
    assert cfg["phase"] == 2
    assert cfg["dwell_ms"] == 500
    assert cfg["camera_index"] == 0  # default backfilled
    with open(os.path.join(str(tmp_path), "config.json")) as fh:
        assert json.load(fh)["phase"] == 2
