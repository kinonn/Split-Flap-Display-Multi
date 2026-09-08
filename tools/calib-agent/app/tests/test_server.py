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


def test_photos_404_before_any_run(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    # run_dir is empty pre-run: must not resolve against the server CWD.
    assert client.get("/api/photos/README.md").status_code == 404


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

        def __init__(self, index=0, brightness=50.0):
            self.brightness = brightness
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
    client = TestClient(server.app)
    client.post("/api/config", json={"display_host": "x",
                                     "llm_base_url": "http://localhost:11434/v1"})
    assert client.post("/api/run/start").status_code == 200
    for _ in range(100):
        if server.harness.state()["status"] == "failed":
            break
        time.sleep(0.02)
    # Old run output is gone; the fresh run reuses the run-001 name
    # (its dir exists again but is empty).
    assert not (old / "photo_0001.jpg").exists()
    assert not (old / "agent_report.json").exists()
    assert server.harness.run_dir.endswith("run-001")
    assert os.listdir(server.harness.run_dir) == []
