"""Server config + read-test endpoints (no hardware)."""

import pytest
from fastapi.testclient import TestClient

import calib_vlm.server as server

from tests.fixtures import FakeCamera, FakeDisplay, SimReader


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_VLM_DATA", str(tmp_path))
    return TestClient(server.app)


def test_config_masks_key(client):
    r = client.post("/api/config", json={"display_host": "x.local",
                                         "llm_api_key": "sk-supersecret",
                                         "camera_exposure": "auto"})
    assert r.status_code == 200
    assert r.json()["llm_api_key"].endswith("cret")
    got = client.get("/api/config").json()
    assert got["display_host"] == "x.local"
    assert got["llm_api_key"] != "sk-supersecret"


def test_config_validates_mode_and_exposure(client):
    assert client.post("/api/config", json={"camera_exposure": "nope"}).status_code == 400
    assert client.post("/api/config", json={"mode": "sideways"}).status_code == 400


def test_config_persists_run_options(client):
    r = client.post("/api/config", json={"mode": "preview", "exhaustive": True,
                                         "min_confidence": 0.75,
                                         "dwell_ms": 500})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "preview"
    assert body["exhaustive"] is True
    assert body["min_confidence"] == pytest.approx(0.75)
    assert body["dwell_ms"] == 500


def test_read_test_roundtrip(client, monkeypatch):
    display = FakeDisplay(total=4)
    monkeypatch.setattr(server, "Display", lambda host: display)
    monkeypatch.setattr(server, "Camera", lambda *a, **k: FakeCamera())
    monkeypatch.setattr(server, "_reader_from_config",
                        lambda cfg: SimReader(display))
    r = client.post("/api/read-test", json={"frame": "AB"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["frame"] == "AB  "     # padded to the module count
    assert body["read"] == "AB  "
    assert body["photo"] == "/api/read-test/photo"


def test_run_start_requires_provider_config(client):
    # No llm_base_url/model configured by default? The defaults ARE set,
    # but no API key for a remote URL -> 400 until configured.
    r = client.post("/api/run/start")
    assert r.status_code == 400
