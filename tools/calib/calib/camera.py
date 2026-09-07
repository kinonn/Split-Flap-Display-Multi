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
    # Drift gate: mean frame-to-frame abs diff (0..255) on a static scene
    # must stay at or below this. check_camera() enforces it; open() uses
    # it to decide when warm-up has settled.
    DRIFT_OK = 2.0
    # Warm-up budget: keep discarding frames until consecutive frames agree
    # (or this expires). The old fixed 0.5 s often ended mid-hunt on
    # backends that ignore the exposure lock, so the check sampled the
    # auto-exposure transient.
    WARMUP_SETTLE_S = 5.0
    # Leading frames of each check batch discarded as settling time.
    CHECK_SETTLE_DISCARD = 4
    # Measurement attempts per check_camera() call: one unlucky window no
    # longer fails the whole check.
    CHECK_ATTEMPTS = 3

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.index = index
        self.width = width
        self.height = height
        self.cap: cv2.VideoCapture | None = None
        self.backend = "unknown"

    @staticmethod
    def _backend_name(value) -> str:
        """Human-readable backend label for diagnostics (unknown-safe)."""
        labels = {}
        for attr in ("CAP_V4L2", "CAP_DSHOW", "CAP_MSMF",
                     "CAP_AVFOUNDATION", "CAP_ANY"):
            if hasattr(cv2, attr):
                try:
                    labels[int(getattr(cv2, attr))] = attr.replace("CAP_", "")
                except (TypeError, ValueError):
                    pass
        try:
            return labels.get(int(value), f"backend-{value}")
        except (TypeError, ValueError):
            return "unknown"

    def _lock_exposure(self) -> None:
        """Best-effort exposure lock, per backend.

        V4L2 uses 3=auto/1=manual while DSHOW/MSMF use 0.75=auto/0.25=
        manual, so the old bare set(AUTO_EXPOSURE, 1.0) was manual on
        Linux but meaningless on Windows — the camera kept hunting and
        check_camera() failed with drift ~2.7. Lock the camera's
        *current* exposure instead of forcing a magic value; where the
        backend refuses (set() returns False), leave auto on and let
        warm-up + check retries cope.
        """
        cap = self.cap
        assert cap is not None
        try:
            current = cap.get(cv2.CAP_PROP_EXPOSURE)
        except Exception:
            current = None
        if self.backend == "V4L2":
            auto, manual = 3.0, 1.0
        else:
            auto, manual = 0.75, 0.25
        try:
            if self.backend == "V4L2":
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto)
            locked = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual)
        except Exception:
            return
        if not locked:
            return
        try:
            exposure = float(current)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        if exposure == 0.0:
            return  # placeholder read; forcing 0 could black out the image
        try:
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        except Exception:
            pass

    def _warmup(self) -> None:
        """Discard frames until the picture settles (or budget expires).

        Minimum 5 reads (first frames are often dark/partial), then keep
        going while consecutive gray frames disagree by more than
        DRIFT_OK.
        """
        cap = self.cap
        assert cap is not None
        prev = None
        reads = 0
        deadline = time.monotonic() + self.WARMUP_SETTLE_S
        while True:
            try:
                ok, frame = cap.read()
            except Exception:
                ok, frame = False, None
            reads += 1
            gray = None
            if ok and frame is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                except Exception:
                    gray = None
            if gray is not None and prev is not None and reads > 5:
                try:
                    settled = float(np.mean(np.abs(gray - prev))) < self.DRIFT_OK
                except Exception:
                    settled = False
                if settled:
                    return
            if gray is not None:
                prev = gray
            if reads > 5 and time.monotonic() > deadline:
                return

    def open(self, quick: bool = False) -> "Camera":
        """Open the camera and prepare it for capture.

        With quick=False (runs, checks) this locks exposure, disables
        autofocus and warms up until frames settle. With quick=True (UI
        live view) it skips the settle wait and just grabs a few frames —
        a slightly unconverged first frame is fine for framing.
        """
        if hasattr(cv2, "CAP_V4L2"):
            cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
            if not cap.isOpened():  # fall back to default backend (macOS/Windows)
                cap = cv2.VideoCapture(self.index)
        else:
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise CameraError(f"cannot open camera index {self.index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap = cap
        try:
            self.backend = self._backend_name(cap.get(cv2.CAP_PROP_BACKEND))
        except Exception:
            self.backend = "unknown"
        self._lock_exposure()
        # Best-effort autofocus off: focus hunting looks exactly like drift.
        try:
            if hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0.0)
        except Exception:
            pass
        if quick:
            for _ in range(3):
                try:
                    cap.read()
                except Exception:
                    pass
            return self
        self._warmup()
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

        The first CHECK_SETTLE_DISCARD frames of each batch are discarded
        (the camera may still be converging when the check starts), drift
        is measured on the trailing `stable_frames`, and the measurement
        retries up to CHECK_ATTEMPTS times before giving up. Returns
        diagnostics; raises CameraError when unusable.
        """
        drifts: list[float] = []
        diag: dict = {}
        problems: list[str] = []
        for attempt in range(1, self.CHECK_ATTEMPTS + 1):
            frames = [self.capture()
                      for _ in range(self.CHECK_SETTLE_DISCARD + stable_frames)]
            frames = frames[self.CHECK_SETTLE_DISCARD:]
            grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
            mean = float(np.mean(grays[-1]))
            # Frame-to-frame drift with a static scene must be small.
            diffs = [float(np.mean(np.abs(grays[i] - grays[i - 1]))) for i in range(1, len(grays))]
            drift = float(np.mean(diffs))
            drifts.append(drift)
            saturated = float(np.mean(grays[-1] >= 250))
            dark = float(np.mean(grays[-1] <= 5))
            diag = {
                "resolution": [int(frames[-1].shape[1]), int(frames[-1].shape[0])],
                "mean_brightness": round(mean, 1),
                "drift": round(drift, 3),
                "saturated_frac": round(saturated, 4),
                "dark_frac": round(dark, 4),
                "backend": self.backend,
                "attempts": attempt,
                "drift_history": [round(d, 3) for d in drifts],
            }
            problems = []
            if not 30 <= mean <= 225:
                problems.append(f"mean brightness {mean:.0f} outside 30..225 (fix lighting/exposure)")
            if drift > self.DRIFT_OK:
                problems.append(f"frame drift {drift:.2f} too high (camera moving or auto-exposure hunting)")
            if saturated > 0.05:
                problems.append("over 5% pixels saturated (reduce exposure/light)")
            if dark > 0.5:
                problems.append("over 50% pixels near black (display out of frame or no light)")
            if not problems:
                return diag
        raise CameraError("; ".join(problems) + f" (after {self.CHECK_ATTEMPTS} attempts)")
