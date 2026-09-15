"""Server config + read-test endpoints (no hardware)."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

import calib_vlm.server as server

from tests.fixtures import FakeCamera, FakeDisplay, SimReader


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_VLM_DATA", str(tmp_path))
    return TestClient(server.app)


@pytest.fixture()
def idle_harness(monkeypatch):
    """Module-level harness in a pristine idle state for a real run."""
    monkeypatch.setattr(server.harness, "status", "idle")
    monkeypatch.setattr(server.harness, "calibrator", None)
    monkeypatch.setattr(server.harness, "pending_abort", False)
    yield server.harness


def _wait_done(client, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get("/api/run/state").json()
        if state["status"] in ("done", "failed", "idle"):
            return state
        time.sleep(0.02)
    raise AssertionError(f"run did not finish: {state}")


def _fake_run(monkeypatch, display):
    """Wire the run thread to the fakes (no camera, display or VLM)."""
    monkeypatch.setattr(server, "_reader_from_config", lambda cfg: None)
    monkeypatch.setattr(server, "Display", lambda host: display)
    monkeypatch.setattr(server, "Camera", lambda *a, **k: FakeCamera())
    monkeypatch.setattr(server, "VLMClient", lambda *a, **k: object())
    monkeypatch.setattr(server, "VlmReader",
                        lambda *a, **k: SimReader(display))


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


def test_config_clamps_declared_ranges(client):
    # kinonn-bot#47: save_config unpacked the declared ranges and never
    # applied them, so dwell_ms > 10000 made the firmware reject every
    # show (HTTP 400) and a negative dwell_ms would reach time.sleep.
    body = client.post("/api/config",
                       json={"dwell_ms": 999999, "timeout_s": 0.1,
                             "min_confidence": 7,
                             "max_seconds": 5}).json()
    assert body["dwell_ms"] == 10000
    assert body["timeout_s"] == 1
    assert body["min_confidence"] == 1
    assert body["max_seconds"] == 60
    low = client.post("/api/config",
                      json={"dwell_ms": -5, "timeout_s": 99999,
                            "min_confidence": -3,
                            "max_seconds": 10 ** 9}).json()
    assert low["dwell_ms"] == 0
    assert low["timeout_s"] == 3600
    assert low["min_confidence"] == 0
    assert low["max_seconds"] == 36000
    # The persisted config is what the run thread reads, and the clamped
    # dwell is inside the firmware's accepted range (the fixture enforces
    # 0..10000 exactly like /api/calib/show does).
    cfg = server.load_config()
    assert (cfg["dwell_ms"], cfg["timeout_s"]) == (0, 3600)
    FakeDisplay(total=4).show_and_settle("H   ", cfg["dwell_ms"])


def test_out_of_range_dwell_still_runs(client, monkeypatch, idle_harness):
    # A stored dwell_ms of -1 (or > 10 s) aborted the run at its first
    # show; the clamp makes the configured value runnable.
    display = FakeDisplay(total=4)
    _fake_run(monkeypatch, display)
    body = client.post("/api/config",
                       json={"dwell_ms": -1,
                             "llm_api_key": "«redacted:sk-…»"}).json()
    assert body["dwell_ms"] == 0
    r = client.post("/api/run/start")
    assert r.status_code == 200, r.text
    state = _wait_done(client)
    assert state["status"] == "done"
    assert "HTTP 400" not in (state["report"] or {}).get("reason", "")
    assert display.fid > 0  # the run really showed frames


def test_run_start_rejected_while_aborting(client, monkeypatch, idle_harness):
    # kinonn-bot#47: start() only rejected status == "running", so a new
    # run could start while the previous one was still aborting, sharing
    # its events/run_dir and releasing its display hold.
    monkeypatch.setattr(server, "_reader_from_config", lambda cfg: None)
    idle_harness.status = "aborting"
    run_seq = idle_harness.run_seq
    r = client.post("/api/run/start")
    assert r.status_code == 409, r.text
    assert idle_harness.state()["status"] == "aborting"
    assert idle_harness.run_seq == run_seq  # no run was started


def test_abort_before_the_calibrator_exists_is_applied(client, monkeypatch,
                                                       idle_harness):
    # kinonn-bot#47: the calibrator is built inside the run thread after
    # the display probe, the camera open and the camera check, but abort()
    # only forwarded when self.calibrator existed — an abort in that
    # window was lost while the status said "aborting".
    entered = threading.Event()
    release = threading.Event()
    made: list = []

    class SlowCalib:
        def __init__(self, *args, **kwargs):
            self.aborted = False
            self.frames_used = 0
            self.vlm_calls = 0
            self.report = {"result": "needs-human",
                           "reason": "aborted by user"}
            made.append(self)
            entered.set()
            release.wait(10)

        def abort(self):
            self.aborted = True

        def run(self):
            return dict(self.report)

    _fake_run(monkeypatch, FakeDisplay(total=4))
    monkeypatch.setattr(server, "VlmCalibrator", SlowCalib)
    client.post("/api/config", json={"llm_api_key": "«redacted:sk-…»"})
    r = client.post("/api/run/start")
    assert r.status_code == 200, r.text
    assert entered.wait(10), "run thread never reached the calibrator"
    assert server.harness.calibrator is None  # the window is open
    r = client.post("/api/run/abort")
    assert r.json()["status"] == "aborting"
    assert made[0].aborted is False  # not constructed when abort arrived
    release.set()
    state = _wait_done(client)
    assert state["status"] == "done"
    assert made[0].aborted is True  # the queued abort was applied


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


def test_photo_endpoint_rejects_separator_names(client):
    # A photo name is a bare file name: on Windows a backslash is a
    # separator too, so "a\..\..\x.png" must not slip past a "/"-only
    # check (os.path.join resolves it outside the run dir there).
    # %5C is the encoded backslash.
    for name in ("a%5C..%5C..%5Csecret.png", "a%5Cb.png"):
        r = client.get(f"/api/photos/{name}")
        assert r.status_code == 400, (name, r.status_code, r.text)
    # A separator-free name is still a normal lookup (404 when missing),
    # never a 400.
    assert client.get("/api/photos/nope.png").status_code == 404
