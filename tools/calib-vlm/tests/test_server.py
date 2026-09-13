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
    assert client.post("/api/config", json={"camera_warmup_s": "nope"}).status_code == 400


def test_config_persists_run_options(client):
    r = client.post("/api/config", json={"mode": "dry-run", "exhaustive": True,
                                         "min_confidence": 0.75,
                                         "dwell_ms": 500,
                                         "camera_warmup_s": 12,
                                         "skip_chars": "AB",
                                         "skip_enabled": False})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "dry-run"
    assert body["exhaustive"] is True
    assert body["min_confidence"] == pytest.approx(0.75)
    assert body["dwell_ms"] == 500
    assert body["skip_chars"] == "AB"
    assert body["skip_enabled"] is False
    # Start-wait clamps to the slider range 0..30 s.
    assert body["camera_warmup_s"] == 12
    assert client.post("/api/config",
                       json={"camera_warmup_s": 99}).json()["camera_warmup_s"] == 30


def test_config_skip_defaults_enabled(client):
    # Fresh config: exclusions enabled with the punctuation defaults.
    body = client.get("/api/config").json()
    assert body["skip_enabled"] is True
    assert body["skip_chars"] == ".'-"
    # Clearing the list is allowed (skip nothing, still enabled).
    cleared = client.post("/api/config", json={"skip_chars": ""}).json()
    assert cleared["skip_chars"] == ""


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


def test_run_start_phases_subset_and_invalid(client, monkeypatch):
    # Phase selection is per-run: valid subsets start, unknown names 400,
    # and the selection is exposed via /api/run/state.
    monkeypatch.setattr(server, "_reader_from_config", lambda cfg: None)
    started = {}

    def fake_start(cfg):
        started.update(cfg)
        return {"status": "running", "run_dir": "x"}

    monkeypatch.setattr(server.harness, "start", fake_start)
    r = client.post("/api/run/start", json={"phases": ["p1", "p4"]})
    assert r.status_code == 200, r.text
    assert started["phases"] == ["p1", "p4"]
    r = client.post("/api/run/start", json={"phases": ["sideways"]})
    assert r.status_code == 400
    # Empty body = full chain default.
    started.clear()
    r = client.post("/api/run/start")
    assert r.status_code == 200, r.text
    assert "phases" not in started


def test_normalize_phases_defaults_and_rejects():
    assert server.normalize_phases(None) == ["p1", "p2", "p4", "acceptance"]
    assert server.normalize_phases([]) == ["p1", "p2", "p4", "acceptance"]
    assert server.normalize_phases(["p4", "p1", "p1"]) == ["p1", "p4"]
    try:
        server.normalize_phases(["p1", "nope"])
    except ValueError as exc:
        assert "unknown phases" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown phase")


def test_event_log_pages_without_truncation(client):
    # Regression: the UI only ever saw the last 100 events (state slice)
    # of a 500-capped buffer while events.jsonl kept everything. The
    # harness must keep the full log and page it via /api/run/events.
    server.harness.events = []
    server.harness.photos = []
    for i in range(350):
        server.harness.log({"t": "", "kind": "read", "text": f"e{i}",
                            "photo": None})
    state = client.get("/api/run/state").json()
    assert state["event_count"] == 350
    page1 = client.get("/api/run/events?offset=0&limit=200").json()
    assert page1["total"] == 350
    assert page1["offset"] == 0
    assert len(page1["events"]) == 200
    assert page1["events"][0]["text"] == "e0"
    page2 = client.get("/api/run/events?offset=200&limit=200").json()
    assert len(page2["events"]) == 150
    assert page2["events"][0]["text"] == "e200"
    assert page2["events"][-1]["text"] == "e349"
