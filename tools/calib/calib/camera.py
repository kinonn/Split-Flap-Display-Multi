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


# Symmetric top/bottom crop bounds: percent of frame height removed from
# EACH end (0 = off). The UI slider spans this range. Module-level (not
# class attributes) so server code can reference them even when the
# Camera class itself is monkeypatched out in tests.
CROP_MIN = 0.0
CROP_MAX = 40.0
DEFAULT_CROP_PERCENT = 30.0


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
    # Sleep between check_camera() attempts (they share one open handle).
    # Some drivers deliver valid-but-all-black frames for a few seconds
    # after open (stream starting / AE not converged); spacing the
    # attempts gives the device total patience instead of burning all
    # attempts back to back on the black startup window.
    CHECK_ATTEMPT_GAP_S = 2.0
    # A stream is "still black" while its mean stays at or below this
    # (raw, pre-brightness-gain). All-black frames are trivially static,
    # so warm-up must never settle on them, and the exposure search must
    # not flip to manual on them (on AE-with-gain drivers manual caps
    # DARKER than the AE result).
    BLACK_MEAN_MIN = 8.0
    BLACK_STREAM_MEAN = 5.0
    BLACK_STREAM_WAIT_S = 1.5
    BLACK_STREAM_POLL_S = 0.25
    # Opening can transiently fail right after another handle released
    # the device (Windows MSMF reports "busy" for a short while), so
    # open() retries a few times before declaring the camera gone.
    OPEN_ATTEMPTS = 3
    OPEN_RETRY_GAP_S = 0.4
    # Auto-exposure search (full open only): walk the exposure property
    # until the frame mean is comfortably inside the check's gates, so
    # warm-up settles in seconds instead of fighting an AE hunt for its
    # whole budget. Closed loop ("keep what improved") because exposure
    # property units differ per backend (log2-seconds on MSMF, raw
    # counts on V4L2); bounded by steps and time so checks stay fast.
    AUTO_BAND = (60.0, 180.0)
    AUTO_MAX_STEPS = 6
    AUTO_BUDGET_S = 4.0
    AUTO_STEP = 1.0

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
                 brightness: float = 50.0, exposure: float | None = None,
                 crop_percent: float = 0.0, warmup_s: float | None = None):
        self.index = index
        self.width = width
        self.height = height
        # Per-instance warm-up budget (seconds), for cameras that hand
        # back all-black frames for a while after open (start-wait
        # slider). None = class default (WARMUP_SETTLE_S); out-of-range
        # or garbage values clamp to 0..30 (never crash a run).
        try:
            self.warmup_s = (None if warmup_s is None
                             else max(0.0, min(30.0, float(warmup_s))))
        except (TypeError, ValueError):
            self.warmup_s = None
        # UI slider scale 0..100, 50 = neutral. Applied in software to
        # every frame capture() delivers: the driver property
        # (CAP_PROP_BRIGHTNESS) is unreliable — many backends ignore it
        # or map it to a narrow range — and the live view must show
        # exactly what the camera check and calibration score.
        self.brightness = brightness
        # Manual sensor exposure (CAP_PROP_EXPOSURE units, backend-
        # dependent). None = auto-search on full open (default); a value
        # is applied directly instead — the fix for a dark live view,
        # where software gain alone only amplifies noise.
        self.exposure = exposure
        # Symmetric top/bottom crop, percent of frame height removed
        # from EACH end (0..40, 0 = off). Applied to the raw frame
        # BEFORE brightness gain, so auto-exposure metering, the camera
        # check, runs and the live view all see the same cropped image.
        self.crop_percent = crop_percent
        self.cap: cv2.VideoCapture | None = None
        self.backend = "unknown"
        self.auto_exposure: dict | None = None

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

    @staticmethod
    def _try_set(cap, prop, value) -> bool:
        """cap.set() that never throws: some drivers (seen on Windows)
        raise cv2.error instead of returning False for properties they
        reject, which used to crash open() with a 500."""
        try:
            return bool(cap.set(prop, value))
        except Exception:
            return False

    def _auto_mode(self) -> float:
        """Backend's auto-mode value for CAP_PROP_AUTO_EXPOSURE."""
        return 3.0 if self.backend == "V4L2" else 0.75

    def _manual_mode(self) -> float:
        """Backend's manual-mode value for CAP_PROP_AUTO_EXPOSURE."""
        return 1.0 if self.backend == "V4L2" else 0.25

    def _lock_exposure(self) -> None:
        """Ensure the driver's auto-exposure is engaged (live view path).

        The old behavior flipped to manual and froze the current value;
        on drivers whose manual range is darker than their AE (AE adds
        gain) that made the live view darker than doing nothing at all.
        """
        cap = self.cap
        assert cap is not None
        self._try_set(cap, cv2.CAP_PROP_AUTO_EXPOSURE, self._auto_mode())

    def _get_exposure(self) -> float | None:
        try:
            value = float(self.cap.get(cv2.CAP_PROP_EXPOSURE))
        except Exception:
            return None
        return None if value == 0.0 else value  # 0.0 = placeholder read

    @property
    def current_exposure(self) -> float | None:
        """The exposure currently set on the device (None if unreadable)."""
        if self.cap is None:
            return None
        return self._get_exposure()

    def _measure(self) -> float | None:
        """One post-gain frame mean, or None when the grab fails."""
        try:
            frame = self.capture(retries=1)
        except Exception:
            return None
        try:
            return float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))
        except Exception:
            return None

    def _manual_mode(self) -> float:
        """Backend's manual-mode value for CAP_PROP_AUTO_EXPOSURE."""
        return 1.0 if self.backend == "V4L2" else 0.25

    def _apply_manual_exposure(self) -> dict:
        """Apply the user-supplied exposure value (no search).

        Returns the diagnostics record: locked when the driver accepted
        both the manual flip and the value, driver_refused otherwise.
        """
        cap = self.cap
        assert cap is not None
        diag: dict = {"steps": 0, "mean_before": None, "mean_after": None,
                      "locked": False, "driver_refused": False,
                      "mode": "manual", "manual": self.exposure}
        self.auto_exposure = diag
        if not self._try_set(cap, cv2.CAP_PROP_AUTO_EXPOSURE,
                             self._manual_mode()):
            diag["driver_refused"] = True
            return diag
        if not self._try_set(cap, cv2.CAP_PROP_EXPOSURE, self.exposure):
            diag["driver_refused"] = True
            return diag
        mean = self._measure()
        if mean is not None:
            diag["mean_before"] = diag["mean_after"] = round(mean, 1)
        diag["locked"] = True
        return diag

    def _auto_exposure(self) -> None:
        """Keep the driver's AE when it lands in the check gates; else
        search manual exposure for better, restoring AE on failure.

        Forcing manual unconditionally (the old behavior) made dark
        scenes DARKER on drivers whose manual range caps above their AE
        (AE adds gain): on the affected camera the mean dropped from 32
        (auto) to 8 (manual max). Full-open only; the live view's quick
        open just ensures AE via _lock_exposure.
        """
        cap = self.cap
        assert cap is not None
        diag: dict = {"steps": 0, "mean_before": None, "mean_after": None,
                      "locked": False, "driver_refused": False, "mode": "auto"}
        self.auto_exposure = diag
        mean = self._measure()
        # A stream can start as valid-but-all-black frames (some drivers
        # need seconds after open). That is not a lighting problem: give
        # it a moment, and never flip to manual on a black stream — on
        # AE-with-gain drivers manual caps DARKER than the AE result.
        deadline = time.monotonic() + self.BLACK_STREAM_WAIT_S
        while (mean is not None and mean <= self.BLACK_STREAM_MEAN
               and time.monotonic() < deadline):
            time.sleep(self.BLACK_STREAM_POLL_S)
            mean = self._measure()
        if mean is None:
            return
        diag["mean_before"] = diag["mean_after"] = round(mean, 1)
        if mean <= self.BLACK_STREAM_MEAN:
            diag["note"] = ("stream still black after open (wrong camera "
                            "index, lens cap, or driver still starting?)")
            return
        if 30.0 <= mean <= 225.0:  # the check's own gates: AE is good enough
            diag["locked"] = True
            return
        if not self._try_set(cap, cv2.CAP_PROP_AUTO_EXPOSURE,
                             self._manual_mode()):
            diag["driver_refused"] = True
            return
        current = self._get_exposure()
        if current is not None:
            # Re-apply so flipping to manual does not jump the image.
            self._try_set(cap, cv2.CAP_PROP_EXPOSURE, current)
        mean = self._measure()
        if mean is None:
            self._try_set(cap, cv2.CAP_PROP_AUTO_EXPOSURE, self._auto_mode())
            return
        lo, hi = self.AUTO_BAND
        deadline = time.monotonic() + self.AUTO_BUDGET_S
        step = self.AUTO_STEP
        for _ in range(self.AUTO_MAX_STEPS):
            if time.monotonic() > deadline:
                break
            current = self._get_exposure()
            if current is None:
                diag["driver_refused"] = True
                break
            moved = False
            for sign in (1, -1):
                if not self._try_set(cap, cv2.CAP_PROP_EXPOSURE,
                                     current + sign * step):
                    continue
                new_mean = self._measure()
                if new_mean is None:
                    self._try_set(cap, cv2.CAP_PROP_EXPOSURE, current)
                    continue
                if abs(new_mean - 128.0) < abs(mean - 128.0):
                    mean = new_mean
                    diag["steps"] += 1
                    moved = True
                    break
                self._try_set(cap, cv2.CAP_PROP_EXPOSURE, current)
            if not moved:
                break
            if lo <= mean <= hi:
                break
            step *= 2
        diag["mean_after"] = round(mean, 1)
        diag["locked"] = 30.0 <= mean <= 225.0
        if diag["locked"]:
            diag["mode"] = "manual"
        else:
            # Manual could not reach the gates either: restore the
            # driver's AE — at least as good as any manual value found.
            self._try_set(cap, cv2.CAP_PROP_AUTO_EXPOSURE, self._auto_mode())
            diag["mode"] = "auto"

    def _warmup(self) -> None:
        """Discard frames until the picture settles (or budget expires).

        Minimum 5 reads (first frames are often dark/partial), then keep
        going while consecutive gray frames disagree by more than
        DRIFT_OK — but a frame only counts as a picture when it is
        brighter than BLACK_MEAN_MIN, because all-black frames are
        trivially static. Raises CameraError when the device opened but
        never produced a single frame, or produced only all-black
        frames (some drivers hand those back for seconds after open) —
        without this open() would "succeed" and the failure resurface
        later as a confusing frame-grab or check error.
        """
        cap = self.cap
        assert cap is not None
        prev = None
        reads = 0
        good = 0
        seen_picture = False  # any frame brighter than BLACK_MEAN_MIN
        # Per-instance override wins (start-wait slider); None = default.
        settle_s = (self.warmup_s if self.warmup_s is not None
                    else self.WARMUP_SETTLE_S)
        deadline = time.monotonic() + settle_s
        while True:
            try:
                ok, frame = cap.read()
            except Exception:
                ok, frame = False, None
            reads += 1
            gray = None
            if ok and frame is not None:
                good += 1
                try:
                    # Settle on the same cropped image every consumer sees.
                    frame = self._apply_crop(frame)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                except Exception:
                    gray = None
            mean = None
            if gray is not None:
                try:
                    mean = float(np.mean(gray))
                except Exception:
                    mean = None
                if mean is not None and mean > self.BLACK_MEAN_MIN:
                    seen_picture = True
            if gray is not None and prev is not None and reads > 5 \
                    and mean is not None:
                try:
                    drift = float(np.mean(np.abs(gray - prev)))
                except Exception:
                    drift = None
                # All-black frames are trivially static: identical black
                # frames must NOT count as settled, or the check below
                # samples a stream that has not produced a picture yet.
                if (drift is not None and drift < self.DRIFT_OK
                        and mean > self.BLACK_MEAN_MIN):
                    return
            if gray is not None:
                prev = gray
            if reads > 5 and time.monotonic() > deadline:
                if good == 0:
                    raise CameraError(
                        "camera opened but returned no frames within "
                        f"{settle_s:.0f}s "
                        "(busy elsewhere or unplugged?)")
                if not seen_picture:
                    raise CameraError(
                        "camera opened but produced only black frames within "
                        f"{settle_s:.0f}s (wrong camera index, lens cap, no "
                        "light, or raise the camera start-wait slider)")
                return

    def _open_backend(self) -> "cv2.VideoCapture | None":
        """One open attempt through the backend preference order.

        Returns an opened capture or None; constructors are guarded
        because some drivers raise cv2.error instead of yielding an
        unopened handle.
        """
        cap = None
        if hasattr(cv2, "CAP_V4L2"):
            try:
                cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
            except Exception:
                cap = None
            if cap is not None and not cap.isOpened():
                # fall back to default backend (macOS/Windows)
                cap.release()
                cap = None
        if cap is None:
            try:
                cap = cv2.VideoCapture(self.index)
            except Exception:
                cap = None
        if cap is not None and not cap.isOpened():
            cap.release()
            return None
        return cap

    def open(self, quick: bool = False) -> "Camera":
        """Open the camera and prepare it for capture.

        With quick=False (runs, checks) this locks exposure, disables
        autofocus and warms up until frames settle. With quick=True (UI
        live view) it skips the settle wait and just grabs a few frames —
        a slightly unconverged first frame is fine for framing.
        """
        cap = None
        for attempt in range(1, self.OPEN_ATTEMPTS + 1):
            cap = self._open_backend()
            if cap is not None:
                break
            if attempt < self.OPEN_ATTEMPTS:
                time.sleep(self.OPEN_RETRY_GAP_S)
        if cap is None:
            raise CameraError(f"cannot open camera index {self.index} "
                              "(busy elsewhere, wrong index, or unplugged?)")
        self._try_set(cap, cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._try_set(cap, cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap = cap
        try:
            self.backend = self._backend_name(cap.get(cv2.CAP_PROP_BACKEND))
        except Exception:
            self.backend = "unknown"
        if quick:
            if self.exposure is None:
                self._lock_exposure()  # live view: cheap lock, no settle wait
            else:
                self._apply_manual_exposure()  # live view with fixed exposure
        else:
            if self.exposure is None:
                self._auto_exposure()  # runs/checks: search, then locked manual
            else:
                self._apply_manual_exposure()  # runs/checks: fixed exposure
        # Best-effort autofocus off: focus hunting looks exactly like drift.
        if hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
            self._try_set(cap, cv2.CAP_PROP_AUTOFOCUS, 0.0)
        if quick:
            for _ in range(3):
                try:
                    cap.read()
                except Exception:
                    pass
            return self
        self._warmup()
        return self

    def _apply_brightness(self, frame: np.ndarray) -> np.ndarray:
        """Software brightness gain applied to every delivered frame.

        Slider 0..100 maps to gain 0..2 with 50 = neutral (1.0), so
        dragging the slider visibly changes the live view on every
        backend. This is the single source of truth for brightness:
        captures (camera check, calibration scoring) see the same image
        the live view shows.
        """
        try:
            b = float(self.brightness)
        except (TypeError, ValueError):
            return frame
        if not np.isfinite(b):
            return frame
        factor = max(0.0, min(100.0, b)) / 50.0
        if abs(factor - 1.0) < 0.01:
            return frame
        return cv2.convertScaleAbs(frame, alpha=factor, beta=0.0)

    def _apply_crop(self, frame: np.ndarray) -> np.ndarray:
        """Symmetric top/bottom crop, percent removed from EACH end.

        Runs on the raw frame BEFORE brightness gain, so the cropped
        pixels never influence auto-exposure metering, the camera
        check, runs, or the live view — every consumer sees the same
        image. Out-of-range/garbage values clamp to CROP_MIN..CROP_MAX
        (never crash a run); a crop that would empty the frame is a
        no-op.
        """
        try:
            pct = float(self.crop_percent)
        except (TypeError, ValueError):
            return frame
        try:
            if not np.isfinite(pct):
                return frame
        except Exception:
            return frame
        pct = max(CROP_MIN, min(CROP_MAX, pct))
        if pct <= 0.0 or frame is None:
            return frame
        try:
            h = int(frame.shape[0])
        except Exception:
            return frame
        if h <= 1:
            return frame
        top = int(h * pct / 100.0)
        if top <= 0 or 2 * top >= h:
            return frame
        try:
            return frame[top:h - top]
        except Exception:
            return frame

    def capture(self, retries: int = 3) -> np.ndarray:
        """Grab one frame, retrying transient driver hiccups.

        Windows drivers commonly return an empty grab when another open
        handle just closed (e.g. the UI live view polling while a check
        runs), so retry briefly before giving up. The top/bottom crop
        applies first, then brightness gain — so exposure metering (via
        _measure) and every consumer see the cropped image.
        """
        if self.cap is None:
            raise CameraError("camera not open")
        last_exc: Exception | None = None
        for _ in range(max(1, retries)):
            try:
                ok, frame = self.cap.read()
            except Exception as exc:  # driver threw instead of returning False
                last_exc = exc
                ok, frame = False, None
            if ok and frame is not None:
                return self._apply_brightness(self._apply_crop(frame))
            time.sleep(0.1)
        detail = f": {last_exc}" if last_exc is not None else " (camera busy elsewhere or unplugged?)"
        raise CameraError(f"frame grab failed{detail}")

    def capture_fresh(self, drain: int = 2, retries: int = 3) -> np.ndarray:
        """Capture after discarding `drain` buffered frames.

        USB drivers buffer frames, so the first grab after a display
        show can be a frame exposed BEFORE the move — the calibration
        then scores the previous frame's text (a systematic one-frame
        lag: every read shows the prior frame). Draining a couple of
        frames before the real capture keeps the photo in sync with
        the just-settled frame. Duck-type tolerant: fakes without a
        raw `cap` just capture().
        """
        cap = getattr(self, "cap", None)
        if cap is None:
            return self.capture(retries=retries)
        for _ in range(max(0, drain)):
            try:
                cap.read()
            except Exception:
                pass
        return self.capture(retries=retries)

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
        retries up to CHECK_ATTEMPTS times (spaced by
        CHECK_ATTEMPT_GAP_S) before giving up. Returns diagnostics;
        raises CameraError when unusable. A large near-black fraction
        (background margins around a lit display strip) is reported in
        the `warnings` key, not failed — the brightness gate here and
        the reader's P0 registration catch a genuinely missing display.
        """
        drifts: list[float] = []
        diag: dict = {}
        problems: list[str] = []
        warnings: list[str] = []
        for attempt in range(1, self.CHECK_ATTEMPTS + 1):
            if attempt > 1:
                # A fresh open can start all-black (drivers take seconds
                # to stream / converge AE): spacing the attempts covers
                # that instead of burning them in the first second.
                time.sleep(self.CHECK_ATTEMPT_GAP_S)
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
            problems = []
            warnings = []
            diag = {
                "resolution": [int(frames[-1].shape[1]), int(frames[-1].shape[0])],
                "brightness": self.brightness,
                "exposure": self.exposure,
                "crop_percent": float(self.crop_percent)
                if isinstance(self.crop_percent, (int, float)) else self.crop_percent,
                "mean_brightness": round(mean, 1),
                "drift": round(drift, 3),
                "saturated_frac": round(saturated, 4),
                "dark_frac": round(dark, 4),
                "backend": self.backend,
                "auto_exposure": dict(self.auto_exposure) if self.auto_exposure else None,
                "attempts": attempt,
                "drift_history": [round(d, 3) for d in drifts],
                "warnings": warnings,
            }
            if not 30 <= mean <= 225:
                problems.append(f"mean brightness {mean:.0f} outside 30..225 (fix lighting/exposure)")
            if drift > self.DRIFT_OK:
                problems.append(f"frame drift {drift:.2f} too high (camera moving or auto-exposure hunting)")
            if saturated > 0.05:
                problems.append("over 5% pixels saturated (reduce exposure/light)")
            # Near-black area is a warning, not a failure: a properly lit
            # display strip in a wide camera view has black margins that
            # the vertical crop cannot remove, and a genuinely missing or
            # unlit display fails the brightness gate here or P0's strict
            # index-strip registration in the calibrator anyway.
            if dark > 0.5:
                warnings.append(f"over 50% pixels near black ({dark:.0%}; "
                                "display out of frame or no light?)")
            if not problems:
                return diag
            if not problems:
                return diag
        raise CameraError("; ".join(problems) + f" (after {self.CHECK_ATTEMPTS} attempts)")
