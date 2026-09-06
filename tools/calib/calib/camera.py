"""USB camera capture with stability/exposure preflight.

Backend is isolated here so Linux/V4L2 can come first and other OS
backends slot in behind the same interface.
"""

from __future__ import annotations

import time

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.index = index
        self.width = width
        self.height = height
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> "Camera":
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():  # fall back to default backend (macOS/Windows)
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise CameraError(f"cannot open camera index {self.index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        # Best-effort exposure lock: auto-calibration needs stable frames.
        # Not all backends honor these; check_camera() verifies stability.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
        self.cap = cap
        time.sleep(0.5)  # let auto-exposure settle before first grab
        # Warm-up reads (first frames are often dark/partial).
        for _ in range(5):
            cap.read()
        return self

    def capture(self) -> np.ndarray:
        if self.cap is None:
            raise CameraError("camera not open")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise CameraError("frame grab failed")
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *args):
        self.close()

    def check_camera(self, stable_frames: int = 10) -> dict:
        """Preflight: grabs frames, checks stability + brightness range.

        Returns diagnostics; raises CameraError when unusable.
        """
        frames = [self.capture() for _ in range(stable_frames)]
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
        mean = float(np.mean(grays[-1]))
        # Frame-to-frame drift with a static scene must be small.
        diffs = [float(np.mean(np.abs(grays[i] - grays[i - 1]))) for i in range(1, len(grays))]
        drift = float(np.mean(diffs))
        saturated = float(np.mean(grays[-1] >= 250))
        dark = float(np.mean(grays[-1] <= 5))
        diag = {
            "resolution": [int(frames[-1].shape[1]), int(frames[-1].shape[0])],
            "mean_brightness": round(mean, 1),
            "drift": round(drift, 3),
            "saturated_frac": round(saturated, 4),
            "dark_frac": round(dark, 4),
        }
        problems = []
        if not 30 <= mean <= 225:
            problems.append(f"mean brightness {mean:.0f} outside 30..225 (fix lighting/exposure)")
        if drift > 2.0:
            problems.append(f"frame drift {drift:.2f} too high (camera moving or auto-exposure hunting)")
        if saturated > 0.05:
            problems.append("over 5% pixels saturated (reduce exposure/light)")
        if dark > 0.5:
            problems.append("over 50% pixels near black (display out of frame or no light)")
        if problems:
            raise CameraError("; ".join(problems))
        return diag
