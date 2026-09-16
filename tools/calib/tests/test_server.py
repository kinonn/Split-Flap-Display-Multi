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
    assert evt["detail"]["took_s"] >= 0  # cycle timing is logged
    assert [s["module"] for s in evt["detail"]["scores"]] == [0, 1, 2, 3]
    assert all(s["verdict"] == "ok" for s in evt["detail"]["scores"])


def test_tune_logs_expected_glyph_and_timing(tmp_path):
    got, sink = _events()
    cal = UICalibrator(FakeDisplay(), FakeCamera(), photo_dir=str(tmp_path),
                       dwell_ms=0, timeout_s=5, max_phase=1, on_event=sink)
    cal.total = 4
    cal.group_widths = [4]  # single local group (normally set from status)
    cal._tune_cell(2, -1, "HHOH")
    kinds = [e["kind"] for e in got]
    # Start line, the cycle's photo, then the outcome line.
    assert kinds == ["tune", "photo", "tune"]
    # Start line names the cell, restates 0-based, shows the expected glyph.
    assert "module 2" in got[0]["text"] and "coarse" in got[0]["text"]
    assert "0-based" in got[0]["text"] and "'O'" in got[0]["text"]
    # Outcome line carries the cycle duration.
    assert "took" in got[1]["text"]


def test_identity_outliers_emit_event(tmp_path):
    got, sink = _events()
    cal = UICalibrator(FakeDisplay(), FakeCamera(), photo_dir=str(tmp_path),
                       dwell_ms=0, timeout_s=5, max_phase=1, on_event=sink)
    cal.total = 4
    rec = cal.shoot("HHHH", "id")
    # No outliers -> no identity event (clean runs stay quiet).
    assert not [e for e in got if e["kind"] == "identity"]
    # Force an outlier through the consensus path.
    cal.consensus(rec, "H")  # flat crops abstain; still no event
    assert not [e for e in got if e["kind"] == "identity"]
    cal._identity_event("consensus", "H", [2])
    evt = [e for e in got if e["kind"] == "identity"][-1]
    assert "m2 (= 3th from left)" in evt["text"]
    assert "0-based" in evt["text"]


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
    # The full event log is persisted in the run folder (JSON lines),
    # not just the last 100 the UI polls.
    with open(os.path.join(h.run_dir, "events.jsonl"), encoding="utf-8") as fh:
        logged = [json.loads(line) for line in fh if line.strip()]
    assert {e["kind"] for e in logged} >= {"run", "phase", "photo"}
    assert len(logged) >= len(st["events"])
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


def test_start_clears_previous_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_DATA", str(tmp_path))
    old = tmp_path / "runs" / "run-001"
    old.mkdir(parents=True)
    (old / "photo_0001.jpg").write_bytes(b"old")
    (old / "report.json").write_text("{}")
    h = Harness()
    h.start(_cfg(), display=FakeDisplay(), camera=FakeCamera())
    h.thread.join(timeout=120)
    st = h.state()
    assert st["status"] == "done", st["report"]
    assert st["run_seq"] == 1  # UI keys "new run" cleanup off this counter
    # Old run output is gone; the fresh run reuses the run-001 name
    # (its dir exists again but only holds this run's files).
    assert st["run_dir"].endswith("run-001")
    assert not (old / "photo_0001.jpg").exists()
    # report.json exists again, but is this run's report, not the stale {}.
    assert (old / "report.json").read_text() != "{}"


def test_photo_rejects_path_traversal():
    with pytest.raises(Exception, match="bad photo name"):
        photo("../report.json")
    with pytest.raises(Exception, match="bad photo name"):
        photo(".hidden")


def test_photo_serves_png_only(tmp_path):
    # Run dirs hold snapshot.json (settings incl. secrets), report.json
    # and events.jsonl beside the photos: only PNGs may be served.
    import calib.server as srv

    run = tmp_path / "run"
    run.mkdir()
    (run / "shot_f1.png").write_bytes(b"fakepng")
    (run / "snapshot.json").write_text('{"settings": {"wifi_psk": "s3cret"}}')
    (run / "report.json").write_text("{}")
    (run / "events.jsonl").write_text("{}\n")
    old = srv.harness.run_dir
    srv.harness.run_dir = str(run)
    try:
        assert photo("shot_f1.png") is not None
        for blocked in ("snapshot.json", "report.json", "events.jsonl", "nope.txt"):
            with pytest.raises(Exception):
                photo(blocked)
    finally:
        srv.harness.run_dir = old


def test_config_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_DATA", str(tmp_path))
    save_config({"display_host": "splitflap.lan", "phase": 2, "dwell_ms": 500})
    cfg = load_config()
    assert cfg["display_host"] == "splitflap.lan"
    assert cfg["phase"] == 2
    assert cfg["dwell_ms"] == 500
    assert cfg["camera_index"] == 0  # default backfilled
    with open(os.path.join(str(tmp_path), "config.json"), encoding="utf-8") as fh:
        assert json.load(fh)["phase"] == 2


def test_config_write_is_atomic_and_private(tmp_path, monkeypatch):
    # Issue kinonn-bot#36: tmp-file + chmod + rename, no leftovers.
    import stat

    from calib.server import _atomic_write_json, config_path

    monkeypatch.setenv("CALIB_DATA", str(tmp_path))
    os.makedirs(str(tmp_path), exist_ok=True)
    _atomic_write_json(config_path(), {"display_host": "h"})
    if os.name != "nt":  # os.chmod cannot express 0o600 on Windows/NTFS
        mode = stat.S_IMODE(os.stat(config_path()).st_mode)
        assert mode == 0o600, oct(mode)
    assert [p for p in os.listdir(str(tmp_path)) if ".tmp-" in p] == []


def test_exposure_config_roundtrip_and_validation(tmp_path, monkeypatch):
    import calib.server as srv

    monkeypatch.setenv("CALIB_DATA", str(tmp_path))
    # Set a value, clear it back to auto, reject garbage with 400.
    srv.save_config({"camera_exposure": -4})
    assert srv.load_config()["camera_exposure"] == -4.0
    srv.save_config({"camera_exposure": ""})
    assert srv.load_config()["camera_exposure"] is None
    with pytest.raises(Exception, match="must be a number"):
        srv.save_config({"camera_exposure": "bright"})


def test_camera_frame_serves_jpeg(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import calib.server as srv

    monkeypatch.setenv("CALIB_DATA", str(tmp_path))

    class FakeCam:
        seen = []

        def __init__(self, index=0, brightness=50.0, exposure=None,
                     crop_percent=15.0):
            self.index = index
            self.brightness = brightness
            self.exposure = exposure
            self.crop_percent = crop_percent
            FakeCam.seen.append(self)

        def open(self, quick=False):
            assert quick  # live view must skip the settle wait
            return self

        def capture(self):
            return np.zeros((48, 64, 3), dtype=np.uint8)

        def close(self):
            pass

    class IdleHarness:
        def state(self):
            return {"status": "idle"}

    monkeypatch.setattr(srv, "Camera", FakeCam)
    monkeypatch.setattr(srv, "harness", IdleHarness())
    client = TestClient(srv.app)
    r = client.get("/api/camera/frame?camera_index=0&brightness=80")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8"
    assert FakeCam.seen[-1].brightness == 80.0
    # Without a query value the saved config applies.
    save_config({"camera_brightness": 33})
    assert client.get("/api/camera/frame").status_code == 200
    assert FakeCam.seen[-1].brightness == 33.0
    # Exposure query value passes through; absent/auto means AE on (None).
    r = client.get("/api/camera/frame?exposure=-4")
    assert r.status_code == 200
    assert FakeCam.seen[-1].exposure == -4.0
    client.get("/api/camera/frame?exposure=auto")
    assert FakeCam.seen[-1].exposure is None
    # Garbage exposure is a 400, not a silent auto.
    assert client.get("/api/camera/frame?exposure=bright").status_code == 400
    # Crop query value passes through (clamped); absent means the saved
    # config (default 30); garbage is a 400.
    r = client.get("/api/camera/frame?crop_percent=10")
    assert r.status_code == 200
    assert FakeCam.seen[-1].crop_percent == 10.0
    r = client.get("/api/camera/frame?crop_percent=99")
    assert r.status_code == 200
    assert FakeCam.seen[-1].crop_percent == 40.0
    save_config({"camera_crop_percent": 5})
    assert client.get("/api/camera/frame").status_code == 200
    assert FakeCam.seen[-1].crop_percent == 5.0
    assert client.get("/api/camera/frame?crop_percent=tall").status_code == 400


def test_crop_config_roundtrip_clamp_and_validation(tmp_path, monkeypatch):
    import calib.server as srv

    monkeypatch.setenv("CALIB_DATA", str(tmp_path))
    # Default backfills to 30 for fresh configs.
    assert srv.load_config()["camera_crop_percent"] == 30.0
    srv.save_config({"camera_crop_percent": 10})
    assert srv.load_config()["camera_crop_percent"] == 10.0
    # Out-of-range clamps to the slider bounds, never stored raw.
    srv.save_config({"camera_crop_percent": 99})
    assert srv.load_config()["camera_crop_percent"] == 40.0
    srv.save_config({"camera_crop_percent": -5})
    assert srv.load_config()["camera_crop_percent"] == 0.0
    # Empty resets to the default; garbage is a 400.
    srv.save_config({"camera_crop_percent": ""})
    assert srv.load_config()["camera_crop_percent"] == 30.0
    with pytest.raises(Exception, match="must be a number"):
        srv.save_config({"camera_crop_percent": "tall"})
    with pytest.raises(Exception, match="must be a number"):
        srv.save_config({"camera_crop_percent": "nan"})


def test_check_camera_accepts_crop_body(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import calib.server as srv

    monkeypatch.setenv("CALIB_DATA", str(tmp_path))

    class FakeCam:
        seen = []

        def __init__(self, index=0, brightness=50.0, exposure=None,
                     crop_percent=15.0):
            self.crop_percent = crop_percent
            FakeCam.seen.append(self)

        def open(self):
            return self

        def check_camera(self):
            return {"crop_percent": self.crop_percent}

        def close(self):
            pass

    class IdleHarness:
        def state(self):
            return {"status": "idle"}

    monkeypatch.setattr(srv, "Camera", FakeCam)
    monkeypatch.setattr(srv, "harness", IdleHarness())
    client = TestClient(srv.app)
    r = client.post("/api/check-camera", json={"camera_crop_percent": 20})
    assert r.status_code == 200
    assert r.json()["diagnostics"]["crop_percent"] == 20.0
    assert FakeCam.seen[-1].crop_percent == 20.0
    # Garbage crop is a 400, not a silent default.
    assert client.post("/api/check-camera",
                       json={"camera_crop_percent": "tall"}).status_code == 400


def test_camera_frame_busy_during_run(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import calib.server as srv

    monkeypatch.setenv("CALIB_DATA", str(tmp_path))

    class BusyHarness:
        def state(self):
            return {"status": "running"}

    monkeypatch.setattr(srv, "harness", BusyHarness())
    assert TestClient(srv.app).get("/api/camera/frame").status_code == 409


def test_check_camera_busy_during_run(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import calib.server as srv

    monkeypatch.setenv("CALIB_DATA", str(tmp_path))

    class BusyHarness:
        def state(self):
            return {"status": "running"}

    monkeypatch.setattr(srv, "harness", BusyHarness())
    assert TestClient(srv.app).post("/api/check-camera", json={}).status_code == 409


def test_quick_endpoints_409_when_camera_lock_held(tmp_path, monkeypatch):
    # A request that slipped past the status guard (e.g. a live-view poll
    # in flight when run/start flipped the status) must be refused
    # instead of double-opening the device: the run opens the camera
    # under _camera_lock, so anyone holding it blocks a second opener.
    from fastapi.testclient import TestClient

    import calib.server as srv

    monkeypatch.setenv("CALIB_DATA", str(tmp_path))
    monkeypatch.setattr(srv, "_CHECK_LOCK_WAIT_S", 0.05)

    class IdleHarness:
        def state(self):
            return {"status": "idle"}

    monkeypatch.setattr(srv, "harness", IdleHarness())
    assert srv._camera_lock.acquire()
    try:
        client = TestClient(srv.app)
        assert client.get("/api/camera/frame").status_code == 409
        assert client.post("/api/check-camera", json={}).status_code == 409
    finally:
        srv._camera_lock.release()


def test_run_fails_cleanly_when_camera_lock_stuck(tmp_path, monkeypatch):
    # A run that cannot get exclusive camera ownership must report a
    # clear needs-human reason — never open the device concurrently.
    import calib.server as srv

    monkeypatch.setenv("CALIB_DATA", str(tmp_path))
    monkeypatch.setattr(srv, "_RUN_LOCK_WAIT_S", 0.2)
    assert srv._camera_lock.acquire()
    try:
        h = Harness()
        h.start(_cfg(), display=FakeDisplay())  # camera=None → run owns it
        h.thread.join(timeout=30)
        st = h.state()
        assert st["status"] == "failed", st["report"]
        assert st["report"]["result"] == "needs-human", st["report"]
        assert "camera" in st["report"]["reason"], st["report"]
    finally:
        srv._camera_lock.release()
