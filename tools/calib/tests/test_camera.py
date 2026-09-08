"""Tests for calib/camera.py with a scripted fake capture device."""

import cv2
import numpy as np
import pytest

from calib.camera import Camera, CameraError


class FakeCapture:
    """Stands in for cv2.VideoCapture: records set() calls, serves a
    scripted per-read gray value sequence (cycled)."""

    backend_value = 200
    exposure_value = -7.0
    set_results = {}
    values = [150]
    fail_reads = 0
    instances = []

    def __init__(self, index, backend=None):
        self.index = index
        self.backend_arg = backend
        self.sets = []
        self.reads = 0
        FakeCapture.instances.append(self)

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_BACKEND:
            return FakeCapture.backend_value
        if prop == cv2.CAP_PROP_EXPOSURE:
            return FakeCapture.exposure_value
        return 0.0

    def set(self, prop, value):
        self.sets.append((prop, value))
        result = FakeCapture.set_results.get(prop, True)
        if isinstance(result, Exception):
            raise result
        return result

    def read(self):
        if FakeCapture.fail_reads > 0:
            FakeCapture.fail_reads -= 1
            return False, None
        value = FakeCapture.values[self.reads % len(FakeCapture.values)]
        self.reads += 1
        return True, np.full((48, 64, 3), value, dtype=np.uint8)

    def release(self):
        pass


@pytest.fixture
def fake_cv(monkeypatch):
    FakeCapture.backend_value = int(cv2.CAP_V4L2)
    FakeCapture.exposure_value = -7.0
    FakeCapture.set_results = {}
    FakeCapture.values = [150]
    FakeCapture.fail_reads = 0
    FakeCapture.instances = []
    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    return FakeCapture


def _sets_of(cap, prop):
    return [value for p, value in cap.sets if p == prop]


def test_v4l2_lock_sequence(fake_cv):
    cam = Camera().open()
    assert cam.backend == "V4L2"
    cap = FakeCapture.instances[-1]
    auto_sets = _sets_of(cap, cv2.CAP_PROP_AUTO_EXPOSURE)
    assert auto_sets[:2] == [3.0, 1.0]  # auto -> manual, then lock current
    assert _sets_of(cap, cv2.CAP_PROP_EXPOSURE) == [-7.0]


def test_dshow_lock_and_set_failure_fallback(fake_cv):
    FakeCapture.backend_value = int(cv2.CAP_DSHOW)
    FakeCapture.set_results = {cv2.CAP_PROP_AUTO_EXPOSURE: False}
    cam = Camera().open()  # backend refusal must not fail the open
    assert cam.backend == "DSHOW"
    cap = FakeCapture.instances[-1]
    assert _sets_of(cap, cv2.CAP_PROP_AUTO_EXPOSURE) == [0.25]
    assert _sets_of(cap, cv2.CAP_PROP_EXPOSURE) == []


def test_autofocus_failure_tolerated(fake_cv):
    FakeCapture.set_results = {cv2.CAP_PROP_AUTOFOCUS: RuntimeError("nope")}
    cam = Camera().open()
    assert cam.cap is not None


def test_throwing_driver_properties_tolerated(fake_cv):
    # Some Windows drivers raise instead of returning False from set().
    FakeCapture.set_results = {
        cv2.CAP_PROP_FRAME_WIDTH: RuntimeError("boom"),
        cv2.CAP_PROP_FRAME_HEIGHT: RuntimeError("boom"),
        cv2.CAP_PROP_BRIGHTNESS: RuntimeError("boom"),
        cv2.CAP_PROP_AUTO_EXPOSURE: RuntimeError("boom"),
        cv2.CAP_PROP_AUTOFOCUS: RuntimeError("boom"),
    }
    cam = Camera().open()
    assert cam.cap is not None
    assert cam.check_camera()["drift"] == 0.0


def test_check_discards_settling_frames(fake_cv):
    # Warm-up consumes the ramp; the check then sees only stable frames.
    FakeCapture.values = [100, 120, 140] + [150] * 30
    cam = Camera().open()
    diag = cam.check_camera()
    assert diag["drift"] == 0.0
    assert diag["attempts"] == 1
    assert diag["backend"] == "V4L2"


def test_check_retries_transient_then_passes(fake_cv):
    FakeCapture.values = [150] * 6 + [150, 160] * 7 + [150] * 14
    cam = Camera().open()
    diag = cam.check_camera()
    assert diag["attempts"] == 2
    assert diag["drift_history"] == [10.0, 0.0]
    assert diag["drift"] == 0.0


def test_check_raises_after_attempts(fake_cv, monkeypatch):
    monkeypatch.setattr(Camera, "WARMUP_SETTLE_S", 0.05)
    FakeCapture.values = [150, 160] * 40
    cam = Camera().open()
    with pytest.raises(CameraError, match=r"frame drift.*after 3 attempts"):
        cam.check_camera()


def test_brightness_applied_and_clamped(fake_cv):
    cam = Camera(brightness=80).open()
    assert cam.brightness == 80
    assert _sets_of(FakeCapture.instances[-1], cv2.CAP_PROP_BRIGHTNESS) == [0.8]
    Camera(brightness=250).open()
    assert _sets_of(FakeCapture.instances[-1], cv2.CAP_PROP_BRIGHTNESS) == [1.0]
    Camera().open()  # default 50 -> 0.5
    assert _sets_of(FakeCapture.instances[-1], cv2.CAP_PROP_BRIGHTNESS) == [0.5]


def test_capture_retries_transient_grab(fake_cv):
    cam = Camera().open()
    FakeCapture.fail_reads = 2
    frame = cam.capture()
    assert frame.shape == (48, 64, 3)
    assert FakeCapture.fail_reads == 0


def test_capture_raises_after_retries(fake_cv):
    cam = Camera().open()
    FakeCapture.fail_reads = 99
    with pytest.raises(CameraError, match="frame grab failed"):
        cam.capture()
