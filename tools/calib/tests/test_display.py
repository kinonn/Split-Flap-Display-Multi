"""Tests for calib/display.py HTTP client with a faked transport."""

import pytest
import requests

from calib.display import CalibError, Display


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeSession:
    """Canned device: busy twice, then idle with an exact-width frame."""

    def __init__(self):
        self.posts = []
        self.polls = 0

    def get(self, url, timeout=None, params=None):
        if url.endswith("/api/calib/status"):
            self.polls += 1
            return FakeResponse({"busy": self.polls < 3, "totalModules": 4,
                                 "contractVersion": 1})
        if url.endswith("/api/calib/frame"):
            return FakeResponse({"settled": True})
        if url.endswith("/calib-contract.json"):
            return FakeResponse({"contractVersion": 1})
        if url.endswith("/settings"):
            return FakeResponse({"settings": {"mode": 0}})
        return FakeResponse({}, status=404)

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))
        if url.endswith("/api/calib/show"):
            return FakeResponse({"frameId": 7, "fleetFrame": False}, status=202)
        return FakeResponse({"message": "ok"})


def _display(monkeypatch):
    monkeypatch.setattr(requests, "Session", FakeSession)
    return Display("splitflap.local")


class PickupGapSession(FakeSession):
    """Queued show not yet picked up: the first polls see an idle display
    with a stale frame, then busy, then idle with the frame settled. The
    old code failed this instantly with 'never reported settled'."""

    def get(self, url, timeout=None, params=None):
        if url.endswith("/api/calib/status"):
            self.polls += 1
            if self.polls <= 2:
                return FakeResponse({"busy": False, "lastFrameId": 6})
            if self.polls <= 4:
                return FakeResponse({"busy": True, "lastFrameId": 6})
            return FakeResponse({"busy": False, "lastFrameId": 7})
        if url.endswith("/api/calib/frame"):
            asked = (params or {}).get("frameId")
            return FakeResponse({"settled": asked == 7 and self.polls >= 5})
        return super().get(url, timeout=timeout, params=params)


def test_show_survives_pickup_gap(monkeypatch):
    monkeypatch.setattr(requests, "Session", PickupGapSession)
    disp = Display("splitflap.local")
    out = disp.show_and_settle("ABCD", dwell_ms=0, timeout_s=5)
    assert out["frameId"] == 7


def test_show_and_settle_polls_until_idle(monkeypatch):
    disp = _display(monkeypatch)
    out = disp.show_and_settle("ABCD", dwell_ms=0, timeout_s=5)
    assert out["frameId"] == 7
    assert disp.session.polls >= 3


def test_hold_snapshot_restore(monkeypatch):
    disp = _display(monkeypatch)
    assert disp.hold(True)["message"] == "ok"
    snap = disp.snapshot()
    assert disp.restore(snap)["message"] == "ok"


def test_http_error_maps_to_calib_error(monkeypatch):
    disp = _display(monkeypatch)
    with pytest.raises(CalibError):
        disp._get("/nope")


class BusyThenOkSession(FakeSession):
    """preview-batch 409s once (display busy), then succeeds."""

    def __init__(self):
        super().__init__()
        self.batch_posts = 0

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))
        if url.endswith("/api/calib/preview-batch"):
            self.batch_posts += 1
            if self.batch_posts == 1:
                return FakeResponse(
                    {"message": "Display busy, poll status until busy==false",
                     "type": "error"}, status=409)
            return FakeResponse({"type": "success", "count": 1}, status=202)
        return super().post(url, json=json, timeout=timeout)

    def get(self, url, timeout=None, params=None):
        if url.endswith("/api/calib/status"):
            self.polls += 1
            # Busy on the first settle poll (right after the 409), then idle.
            return FakeResponse({"busy": self.polls < 2, "totalModules": 4,
                                 "contractVersion": 1})
        return super().get(url, timeout=timeout, params=params)


def test_preview_batch_409_retries_until_idle(monkeypatch):
    # Regression: a single transient 409 (display busy) used to abort the
    # whole run. The firmware's documented contract is back off and poll;
    # the client must retry once the display settles.
    monkeypatch.setattr(requests, "Session", BusyThenOkSession)
    disp = Display("splitflap.local")
    out = disp.preview_batch([{"module": 1, "charIndex": -1, "delta": 5}])
    assert out["type"] == "success"
    assert disp.session.batch_posts == 2


class AlwaysBusySession(BusyThenOkSession):
    """preview-batch 409s forever; the client must give up eventually."""

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))
        if url.endswith("/api/calib/preview-batch"):
            self.batch_posts += 1
            return FakeResponse({"message": "Display busy", "type": "error"},
                                status=409)
        return super().post(url, json=json, timeout=timeout)


def test_preview_batch_persistent_409_raises(monkeypatch):
    monkeypatch.setattr(requests, "Session", AlwaysBusySession)
    disp = Display("splitflap.local", settle_timeout_s=0.5)
    with pytest.raises(CalibError, match=r"HTTP 409"):
        disp.preview_batch([{"module": 1, "charIndex": -1, "delta": 5}])


class ServerErrorSession(FakeSession):
    """preview-batch 500s: a non-409 error must NOT be retried."""

    def __init__(self):
        super().__init__()
        self.batch_posts = 0

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json))
        if url.endswith("/api/calib/preview-batch"):
            self.batch_posts += 1
            return FakeResponse({"message": "boom"}, status=500)
        return super().post(url, json=json, timeout=timeout)


def test_non_409_error_not_retried(monkeypatch):
    monkeypatch.setattr(requests, "Session", ServerErrorSession)
    disp = Display("splitflap.local")
    with pytest.raises(CalibError, match=r"HTTP 500"):
        disp.preview_batch([{"module": 1, "charIndex": -1, "delta": 5}])
    assert disp.session.batch_posts == 1
