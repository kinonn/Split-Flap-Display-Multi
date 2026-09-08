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


def test_brightness_slider_adjusts_captured_frames(fake_cv):
    # Slider 0..100 applies a software gain (50 = neutral, gain 0..2) to
    # every captured frame, so the live view reflects the slider on
    # backends that ignore CAP_PROP_BRIGHTNESS.
    FakeCapture.values = [100]
    cam = Camera(brightness=75).open()
    assert float(np.mean(cam.capture())) == 150.0
    cam = Camera(brightness=25).open()
    assert float(np.mean(cam.capture())) == 50.0
    cam = Camera(brightness=50).open()
    assert float(np.mean(cam.capture())) == 100.0
    # Out-of-range slider values clamp to 0..100 (gain 2 at the top).
    cam = Camera(brightness=250).open()
    assert float(np.mean(cam.capture())) == 200.0
    # No driver CAP_PROP_BRIGHTNESS is set anymore (software-only).
    assert _sets_of(FakeCapture.instances[-1], cv2.CAP_PROP_BRIGHTNESS) == []


def test_brightness_reported_in_check_diagnostics(fake_cv):
    FakeCapture.values = [100]
    cam = Camera(brightness=75).open()
    diag = cam.check_camera()
    assert diag["brightness"] == 75
    # The measured mean reflects the software gain, not the raw frame.
    assert diag["mean_brightness"] == 150.0


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


def test_open_retries_transient_busy(fake_cv, monkeypatch):
    # A device just released by another handle can refuse the first open
    # attempt (Windows MSMF reports "busy" for a moment): open() must
    # retry before giving up.
    monkeypatch.setattr(Camera, "OPEN_ATTEMPTS", 3)
    monkeypatch.setattr(Camera, "OPEN_RETRY_GAP_S", 0.01)
    built = []
    orig_init = FakeCapture.__init__

    def counting_init(self, index, backend=None):
        built.append(self)
        orig_init(self, index, backend)

    monkeypatch.setattr(FakeCapture, "__init__", counting_init)
    # First two constructor attempts yield an unopened handle.
    monkeypatch.setattr(FakeCapture, "isOpened", lambda self: built.index(self) >= 2)
    cam = Camera().open()
    assert len(built) == 3
    assert cam.cap is built[-1]


def test_open_raises_when_always_busy(fake_cv, monkeypatch):
    monkeypatch.setattr(Camera, "OPEN_RETRY_GAP_S", 0.01)
    monkeypatch.setattr(FakeCapture, "isOpened", lambda self: False)
    with pytest.raises(CameraError, match="cannot open camera index 0"):
        Camera().open()


def test_open_raises_when_no_frames_ever_arrive(fake_cv, monkeypatch):
    # Windows can report the device open yet hand back only empty grabs
    # while another process still holds it: fail fast at open with a
    # clear reason instead of a confusing frame-grab error later.
    monkeypatch.setattr(Camera, "WARMUP_SETTLE_S", 0.1)
    FakeCapture.fail_reads = 10 ** 9  # never serves a frame during the test
    with pytest.raises(CameraError, match="returned no frames"):
        Camera().open()
