"""Tests for agent_app/server.py API surface (no display/camera needed)."""

import json
import os
import time

from fastapi.testclient import TestClient

from agent_app import server


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_AGENT_DATA", str(tmp_path))
    monkeypatch.setattr(server.harness, "run_dir", "")
    monkeypatch.setattr(server.harness, "status", "idle")
    return TestClient(server.app)


def test_templates_empty_before_run(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    assert client.get("/api/run/templates").json() == {"glyphs": [], "source": None}


def test_templates_served_from_run_dir(tmp_path, monkeypatch):
    run_dir = tmp_path / "run-001"
    tpl_dir = run_dir / "templates"
    tpl_dir.mkdir(parents=True)
    (tpl_dir / "glyph_U0048.png").write_bytes(b"\x89PNG-fake")
    (tpl_dir / "manifest.json").write_text(json.dumps(
        {"source": "test", "glyphs": {"H": "glyph_U0048.png"}}))
    monkeypatch.setattr(server.harness, "run_dir", str(run_dir))
    client = TestClient(server.app)
    assert client.get("/api/run/templates").json() == {"glyphs": ["H"], "source": "test"}
    assert client.get("/api/run/templates/H").status_code == 200
    assert client.get("/api/run/templates/AB").status_code == 400
    assert client.get("/api/run/templates/Z").status_code == 404


def test_config_masks_key(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_AGENT_DATA", str(tmp_path))
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client = TestClient(server.app)
    client.post("/api/config", json={"display_host": "h", "llm_api_key": "sk-secret123"})
    got = client.get("/api/config").json()
    assert got["llm_api_key"].endswith("123")
    assert "secret" not in got["llm_api_key"]


def test_full_drum_toggle_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_AGENT_DATA", str(tmp_path))
    client = TestClient(server.app)
    assert client.get("/api/config").json()["full_drum"] is False
    client.post("/api/config", json={"full_drum": True})
    assert client.get("/api/config").json()["full_drum"] is True
    client.post("/api/config", json={"full_drum": False})
    assert client.get("/api/config").json()["full_drum"] is False


def test_exposure_config_roundtrip_and_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_AGENT_DATA", str(tmp_path))
    client = TestClient(server.app)
    # Set a value, clear it back to auto, reject garbage with 400.
    assert client.post("/api/config", json={"camera_exposure": -4}).status_code == 200
    assert client.get("/api/config").json()["camera_exposure"] == -4.0
    assert client.post("/api/config", json={"camera_exposure": ""}).status_code == 200
    assert client.get("/api/config").json()["camera_exposure"] is None
    r = client.post("/api/config", json={"camera_exposure": "bright"})
    assert r.status_code == 400


def test_crop_config_roundtrip_clamp_and_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_AGENT_DATA", str(tmp_path))
    client = TestClient(server.app)
    # Default backfills to 30 for fresh configs.
    assert client.get("/api/config").json()["camera_crop_percent"] == 30.0
    assert client.post("/api/config", json={"camera_crop_percent": 10}).status_code == 200
    assert client.get("/api/config").json()["camera_crop_percent"] == 10.0
    # Out-of-range clamps to the slider bounds, never stored raw.
    assert client.post("/api/config", json={"camera_crop_percent": 99}).status_code == 200
    assert client.get("/api/config").json()["camera_crop_percent"] == 40.0
    assert client.post("/api/config", json={"camera_crop_percent": -5}).status_code == 200
    assert client.get("/api/config").json()["camera_crop_percent"] == 0.0
    # Empty resets to the default; garbage is a 400.
    assert client.post("/api/config", json={"camera_crop_percent": ""}).status_code == 200
    assert client.get("/api/config").json()["camera_crop_percent"] == 30.0
    assert client.post("/api/config", json={"camera_crop_percent": "tall"}).status_code == 400
    assert client.post("/api/config", json={"camera_crop_percent": "nan"}).status_code == 400


def test_config_write_is_atomic_and_private(tmp_path, monkeypatch):
    # Issue kinonn-bot#36: tmp-file + chmod + rename, no leftovers.
    import os
    import stat

    from agent_app.server import _atomic_write_json, config_path

    monkeypatch.setenv("CALIB_AGENT_DATA", str(tmp_path))
    _atomic_write_json(config_path(), {"llm_api_key": "sk-secret123"})
    if os.name != "nt":  # os.chmod cannot express 0o600 on Windows/NTFS
        mode = stat.S_IMODE(os.stat(config_path()).st_mode)
        assert mode == 0o600, oct(mode)
    assert json.load(open(config_path())) == {"llm_api_key": "sk-secret123"}
    assert [p for p in os.listdir(tmp_path) if ".tmp-" in p] == []


def test_photos_404_before_any_run(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    # run_dir is empty pre-run: must not resolve against the server CWD.
    assert client.get("/api/photos/README.md").status_code == 404


def test_photos_serve_png_only(tmp_path, monkeypatch):
    # Run dirs hold snapshot.json (settings incl. secrets) beside the
    # photos: the route must never serve non-PNG files.
    client = _client(tmp_path, monkeypatch)
    run = tmp_path / "runs" / "run-001"
    run.mkdir(parents=True)
    (run / "shot_f1.png").write_bytes(b"fakepng")
    (run / "snapshot.json").write_text('{"settings": {"wifi_psk": "s3cret"}}')
    (run / "report.json").write_text("{}")
    (run / "events.jsonl").write_text("{}\n")
    monkeypatch.setattr(server.harness, "run_dir", str(run))
    assert client.get("/api/photos/shot_f1.png").status_code == 200
    assert client.get("/api/photos/snapshot.json").status_code == 404
    assert client.get("/api/photos/report.json").status_code == 404
    assert client.get("/api/photos/events.jsonl").status_code == 404


def test_start_key_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_AGENT_DATA", str(tmp_path))
    for env in ("DISPLAY_HOST", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    client = TestClient(server.app)
    client.post("/api/config", json={"display_host": "x",
                                     "llm_base_url": "https://api.openai.com/v1",
                                     "llm_model": "m"})
    # hosted endpoint without a key -> rejected
    assert client.post("/api/run/start").status_code == 400
    # local endpoint needs no key -> accepted
    client.post("/api/config", json={"llm_base_url": "http://localhost:11434/v1"})
    assert client.post("/api/run/start").status_code == 200


def test_camera_frame_serves_jpeg(tmp_path, monkeypatch):
    import numpy as np

    monkeypatch.setenv("CALIB_AGENT_DATA", str(tmp_path))
    monkeypatch.setattr(server.harness, "run_dir", "")
    monkeypatch.setattr(server.harness, "status", "idle")

    class FakeCam:
        seen = []

        def __init__(self, index=0, brightness=50.0, exposure=None,
                     crop_percent=15.0):
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

    monkeypatch.setattr(server, "Camera", FakeCam)
    client = TestClient(server.app)
    r = client.get("/api/camera/frame?camera_index=0&brightness=80")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8"
    assert FakeCam.seen[-1].brightness == 80.0
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
    assert client.post("/api/config", json={"camera_crop_percent": 5}).status_code == 200
    assert client.get("/api/camera/frame").status_code == 200
    assert FakeCam.seen[-1].crop_percent == 5.0
    assert client.get("/api/camera/frame?crop_percent=tall").status_code == 400


def test_check_camera_accepts_crop_body(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_AGENT_DATA", str(tmp_path))
    monkeypatch.setattr(server.harness, "run_dir", "")
    monkeypatch.setattr(server.harness, "status", "idle")

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

    monkeypatch.setattr(server, "Camera", FakeCam)
    client = TestClient(server.app)
    r = client.post("/api/check-camera", json={"camera_crop_percent": 20})
    assert r.status_code == 200
    assert r.json()["diagnostics"]["crop_percent"] == 20.0
    assert FakeCam.seen[-1].crop_percent == 20.0
    # Garbage crop is a 400, not a silent default.
    assert client.post("/api/check-camera",
                       json={"camera_crop_percent": "tall"}).status_code == 400


def test_camera_frame_busy_during_run(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_AGENT_DATA", str(tmp_path))
    monkeypatch.setattr(server.harness, "run_dir", "")
    monkeypatch.setattr(server.harness, "status", "running")
    client = TestClient(server.app)
    assert client.get("/api/camera/frame").status_code == 409


def test_check_camera_busy_during_run(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_AGENT_DATA", str(tmp_path))
    monkeypatch.setattr(server.harness, "run_dir", "")
    monkeypatch.setattr(server.harness, "status", "running")
    client = TestClient(server.app)
    assert client.post("/api/check-camera", json={}).status_code == 409


def test_quick_endpoints_409_when_camera_lock_held(tmp_path, monkeypatch):
    # A request that slipped past the status guard (e.g. a live-view poll
    # in flight when run/start flipped the status) must be refused
    # instead of double-opening the device: the run opens the camera
    # under _camera_lock, so anyone holding it blocks a second opener.
    monkeypatch.setenv("CALIB_AGENT_DATA", str(tmp_path))
    monkeypatch.setattr(server.harness, "run_dir", "")
    monkeypatch.setattr(server.harness, "status", "idle")
    monkeypatch.setattr(server, "_CHECK_LOCK_WAIT_S", 0.05)
    assert server._camera_lock.acquire()
    try:
        client = TestClient(server.app)
        assert client.get("/api/camera/frame").status_code == 409
        assert client.post("/api/check-camera", json={}).status_code == 409
    finally:
        server._camera_lock.release()


def test_vlm_session_id_stable_per_key():
    a = server.vlm_session_id({"llm_api_key": "k1", "llm_base_url": "u"})
    b = server.vlm_session_id({"llm_api_key": "k1", "llm_base_url": "u"})
    c = server.vlm_session_id({"llm_api_key": "k2", "llm_base_url": "u"})
    d = server.vlm_session_id({"llm_base_url": "u"})  # no key -> url-scoped
    assert a == b  # same key -> same session (warm prompt cache per run)
    assert a != c and a != d


def test_start_clears_previous_runs(tmp_path, monkeypatch):
    from calib.display import CalibError

    monkeypatch.setenv("CALIB_AGENT_DATA", str(tmp_path))
    monkeypatch.setattr(server.harness, "run_dir", "")
    monkeypatch.setattr(server.harness, "status", "idle")
    old = tmp_path / "runs" / "run-001"
    old.mkdir(parents=True)
    (old / "photo_0001.jpg").write_bytes(b"old")
    (old / "agent_report.json").write_text("{}")

    class DeadDisplay:
        def __init__(self, host):
            pass

        def status(self):
            raise CalibError("offline")

    monkeypatch.setattr(server, "Display", DeadDisplay)
    seq_before = server.harness.state()["run_seq"]
    client = TestClient(server.app)
    client.post("/api/config", json={"display_host": "x",
                                     "llm_base_url": "http://localhost:11434/v1"})
    assert client.post("/api/run/start").status_code == 200
    for _ in range(100):
        if server.harness.state()["status"] == "failed":
            break
        time.sleep(0.02)
    # Old run output is gone; the fresh run reuses the run-001 name.
    assert not (old / "photo_0001.jpg").exists()
    assert not (old / "agent_report.json").exists()
    assert server.harness.run_dir.endswith("run-001")
    assert server.harness.state()["run_seq"] == seq_before + 1  # UI keys cleanup off this
    # The run's log is persisted in the run folder (JSON lines) — even a
    # failed run leaves its narrative on disk.
    log_path = os.path.join(server.harness.run_dir, "events.jsonl")
    with open(log_path, encoding="utf-8") as fh:
        logged = [json.loads(line) for line in fh if line.strip()]
    kinds = [e["kind"] for e in logged]
    assert "run" in kinds and "error" in kinds
    assert any("cleared" in e["text"] for e in logged if e["kind"] == "run")
