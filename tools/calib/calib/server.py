"""Deterministic calibration web UI (no LLM).

Browser front-end for the P0->P4 ``Calibrator`` in ``calib/loop.py``:
live event log, photo gallery with per-module verdicts, and the final
report. Mirrors the ``tools/calib-agent/app`` API shape where it makes
sense (``/api/run/state``, ``/api/photos/{name}``, template + snapshot
routes) so the two UIs feel like one — minus everything LLM.

Run:
    uv run splitflap-calib-ui   # http://127.0.0.1:8001
"""

from __future__ import annotations

import json
import math
import os
import shutil
import threading
import time

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

from .camera import (CROP_MAX, CROP_MIN, DEFAULT_CROP_PERCENT, Camera,
                     CameraError)
from .display import CalibError, Display
from .loop import Calibrator

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Camera ownership: the run is the ONLY component that opens the camera
# for real work, and it does so under this lock for the whole run (only
# when it owns the camera; injected fakes in tests skip it). The quick
# endpoints (check, live frame) take it non-blocking/bounded and answer
# 409 when busy — FastAPI runs each request in its own thread and
# Windows webcam drivers handle concurrent opens badly (empty grabs).
_camera_lock = threading.Lock()
# Bounded waits so nobody ever opens the device twice: a live-view poll
# releases it within ~2 s, so the check briefly waits for stragglers and
# the run waits a little longer before giving up.
_CHECK_LOCK_WAIT_S = 2.5
_RUN_LOCK_WAIT_S = 30.0


def data_dir() -> str:
    return os.environ.get("CALIB_DATA", os.path.join(os.getcwd(), "calib-ui-data"))


def alloc_run_dir(runs_dir: str) -> str:
    """Allocate the first unused run-NNN dir (collision-proof).

    The old listdir-count scheme reused numbers after a deletion (and
    counted stray files), silently merging two runs. Probing with
    exist_ok=False also closes the create race.
    """
    os.makedirs(runs_dir, exist_ok=True)
    n = 1
    while True:
        run_dir = os.path.join(runs_dir, f"run-{n:03d}")
        try:
            os.makedirs(run_dir, exist_ok=False)
            return run_dir
        except FileExistsError:
            n += 1


def config_path() -> str:
    return os.path.join(data_dir(), "config.json")


def clear_previous_runs(runs_dir: str) -> int:
    """Delete the output of earlier runs (photos, logs, reports, snapshots).

    Called when a run starts so runs/ only ever holds the active run —
    stale evidence must not linger beside (or be mistaken for) current
    results. Returns how many entries were removed.
    """
    removed = 0
    if os.path.isdir(runs_dir):
        for name in os.listdir(runs_dir):
            path = os.path.join(runs_dir, name)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
                removed += 1
            except OSError:
                pass
    return removed


DEFAULTS = {
    "display_host": "splitflap.local",
    "camera_index": 0,
    "camera_brightness": 50,
    "camera_exposure": None,  # None = driver AE; a value fixes the sensor
    "camera_crop_percent": DEFAULT_CROP_PERCENT,  # % trimmed top AND bottom (0..40)
    "phase": 4,
    "dwell_ms": 800,
    "timeout_s": 60.0,
    "identity_thresh": 0.85,
    "relearn_templates": False,
}


def load_config() -> dict:
    cfg: dict = {}
    try:
        with open(config_path(), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        pass
    for key, value in DEFAULTS.items():
        cfg.setdefault(key, value)
    return cfg


def save_config(patch: dict) -> dict:
    os.makedirs(data_dir(), exist_ok=True)
    try:
        with open(config_path(), encoding="utf-8") as fh:
            stored = json.load(fh)
    except (OSError, ValueError):
        stored = {}
    for key in DEFAULTS:
        if key in patch and patch[key] not in (None, ""):
            if key == "camera_crop_percent":
                continue  # validated + clamped below, not stored raw
            stored[key] = patch[key]
    # Exposure is settable AND clearable (null/"" = back to auto-search).
    # Garbage is a 400, matching _exposure_of — never a 500, never silent.
    if "camera_exposure" in patch:
        raw = patch["camera_exposure"]
        if raw in (None, ""):
            stored["camera_exposure"] = None
        else:
            try:
                stored["camera_exposure"] = float(raw)
            except (TypeError, ValueError):
                raise HTTPException(400, "camera_exposure must be a number or empty (auto)")
    # Crop is validated + clamped to the slider range (0..40, default 30).
    # Garbage (incl. NaN/inf) is a 400, matching _crop_of — never a 500,
    # never silent.
    if "camera_crop_percent" in patch:
        raw = patch["camera_crop_percent"]
        if raw in (None, ""):
            stored["camera_crop_percent"] = DEFAULT_CROP_PERCENT
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise HTTPException(400, "camera_crop_percent must be a number 0..40")
            if not math.isfinite(value):
                raise HTTPException(400, "camera_crop_percent must be a number 0..40")
            stored["camera_crop_percent"] = max(
                CROP_MIN, min(CROP_MAX, value))
    _atomic_write_json(config_path(), stored)
    return load_config()


def _atomic_write_json(path: str, payload: dict) -> None:
    """Write JSON with mode 0600, atomically (issue kinonn-bot#36).

    The old write-then-chmod left a window where the file was
    world-readable, and a mid-write crash left a truncated config.
    Writing to a same-dir temp file, chmodding it, then os.replace()
    closes both windows. The temp name carries pid + thread id: the sync
    /api/config endpoints run on a threadpool, so two concurrent saves
    must not share one temp file.
    """
    tmp = path + f".tmp-{os.getpid()}-{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


class UICalibrator(Calibrator):
    """Calibrator that streams progress to a UI event sink.

    No calibration logic lives here — every override only logs around
    the ``super()`` call. Abort is cooperative: the harness flips the
    flag and the next ``shoot()``/settle poll raises, which
    ``Calibrator.run()`` already turns into a ``needs-human`` report
    with rollback.
    """

    def __init__(self, *args, on_event=None, abort_flag=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_event = on_event or (lambda e: None)
        self.abort_flag = abort_flag or (lambda: False)

    def _abort_requested(self) -> bool:
        try:
            return bool(self.abort_flag())
        except Exception:
            return False

    def event(self, kind: str, text: str, photo: str | None = None,
              detail: dict | None = None):
        evt: dict = {"t": time.strftime("%H:%M:%S"), "kind": kind,
                     "text": text, "photo": photo}
        if detail is not None:
            evt["detail"] = detail
        self.on_event(evt)

    def shoot(self, frame: str, tag: str) -> dict:
        if self.abort_flag():
            raise CalibError("aborted by user")
        t0 = time.monotonic()
        rec = super().shoot(frame, tag)
        scores = self.scores(rec)
        took = time.monotonic() - t0
        bad = [(i, s["verdict"]) for i, s in enumerate(scores)
               if s["verdict"] != "ok"]
        summary = f"{len(scores) - len(bad)}/{len(scores)} ok" if scores else "no crops"
        if bad:
            summary += f" — off: {', '.join(f'm{i} {v}' for i, v in bad)}"
        self.event("photo", f"{tag} frameId={rec['frameId']} "
                            f"show={frame!r} → {summary} (took {took:.1f}s)",
                   photo=os.path.basename(rec["photo"]),
                   detail={"frame": frame, "frameId": rec["frameId"], "tag": tag,
                           "took_s": round(took, 1),
                           "scores": [{"module": i, **s}
                                      for i, s in enumerate(scores)]})
        return rec

    @staticmethod
    def _glyph_at(frame: str, module: int) -> str:
        """Expected glyph at a global module index (0-based, photo order)."""
        return frame[module] if 0 <= module < len(frame) else "?"

    def _tune_cell(self, module: int, char_index: int, show_frame: str) -> dict:
        cell = "coarse" if char_index < 0 else f"char {char_index}"
        self.event("tune", f"tuning module {module} {cell} "
                           f"(0-based; expected glyph "
                           f"{self._glyph_at(show_frame, module)!r} "
                           "at this module) …")
        t0 = time.monotonic()
        out = super()._tune_cell(module, char_index, show_frame)
        took = time.monotonic() - t0
        last = self.deltas[-1] if self.deltas else {}
        if last.get("proposal"):
            self.event("tune", f"module {module}: proposal only "
                               f"(phase 1/2, hardware untouched; took {took:.1f}s)")
        elif last.get("new") != last.get("old"):
            self.event("tune", f"module {module}: {last.get('old')} → "
                               f"{last.get('new')} (kept; took {took:.1f}s)")
        else:
            self.event("tune", f"module {module}: no improvement, kept "
                               f"{last.get('old')} (took {took:.1f}s)")
        return out

    def _identity_event(self, method: str, glyph: str, outliers: list[int]):
        """Surface wrong-glyph findings (silent until now: they only
        landed in the final report). 0-based, spelled out like the
        agent app's suspects field."""
        if not outliers:
            return
        spots = ", ".join(f"m{i} (= {i + 1}th from left)" for i in outliers)
        self.event("identity", f"wrong glyph ({method}) on frame {glyph!r}: "
                               f"{spots} — 0-based indices")

    def consensus(self, rec: dict, glyph: str) -> list[int]:
        out = super().consensus(rec, glyph)
        self._identity_event("consensus", glyph, out)
        return out

    def absolute(self, rec: dict, glyph: str) -> list[int]:
        out = super().absolute(rec, glyph)
        self._identity_event("template", glyph, out)
        return out

    def _phase(self, name: str, label: str, fn):
        self.event("phase", f"{name}: {label} — started")
        out = fn()
        self.event("phase", f"{name}: {label} — done")
        return out

    def _p0_register(self):
        return self._phase("P0", "registration", super()._p0_register)

    def _p1_coarse(self):
        return self._phase("P1", "coarse sweep", super()._p1_coarse)

    def _p2_fine(self):
        return self._phase("P2", "fine sweep", super()._p2_fine)

    def _p3_boundaries(self):
        return self._phase("P3", "boundary pairs", super()._p3_boundaries)

    def _p4_repeatability(self):
        return self._phase("P4", "repeatability", super()._p4_repeatability)


class Harness:
    """Owns one deterministic run at a time (background thread)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.photos: list[str] = []
        self.status = "idle"
        self.phase = 4
        self.report: dict | None = None
        self.run_dir = ""
        self.calib: UICalibrator | None = None
        self.thread: threading.Thread | None = None
        self.abort_requested = False
        self.run_seq = 0  # increments per start; lets the UI detect a new run

    def log(self, event: dict):
        with self.lock:
            self.events.append(event)
            if event.get("photo") and event["photo"] not in self.photos:
                self.photos.append(event["photo"])
            self.events = self.events[-500:]
            run_dir = self.run_dir
        # Persist the full log as JSON lines in the run folder: the UI
        # only ever sees the last 100 events, and everything in memory
        # dies with the server. Appends are cheap at log frequency.
        if run_dir:
            try:
                with open(os.path.join(run_dir, "events.jsonl"), "a",
                          encoding="utf-8") as fh:
                    fh.write(json.dumps(event, default=str) + "\n")
            except OSError:
                pass  # logging must never break a run

    def state(self) -> dict:
        with self.lock:
            calib = self.calib
            return {"status": self.status, "events": self.events[-100:],
                    "photos": self.photos[-24:], "report": self.report,
                    "run_dir": self.run_dir, "phase": self.phase,
                    "run_seq": self.run_seq,
                    "previews": calib.previews if calib else 0,
                    "persists": calib.persists if calib else 0,
                    "sweeps": round(calib.sweeps, 2) if calib else 0}

    def start(self, cfg: dict, display=None, camera=None) -> dict:
        """Start a run. ``display``/``camera`` inject fakes (tests only)."""
        with self.lock:
            if self.status == "running":
                raise HTTPException(409, "run already in progress")
            self.status = "running"
            self.phase = int(cfg.get("phase", 4))
            self.events = []
            self.photos = []
            self.report = None
            self.calib = None
            self.run_seq += 1
            self.abort_requested = False
        runs_dir = os.path.join(data_dir(), "runs")
        cleared = clear_previous_runs(runs_dir)
        run_dir = alloc_run_dir(runs_dir)
        with self.lock:
            self.run_dir = run_dir
        if cleared:
            self.log({"t": time.strftime("%H:%M:%S"), "kind": "run",
                      "text": f"cleared {cleared} previous run(s) "
                              "(old logs, photos, reports)", "photo": None})

        def _run():
            cam = camera
            owns_camera = cam is None
            cam_lock_held = False
            try:
                disp = display or Display(cfg["display_host"],
                                         timeout_s=float(cfg.get("timeout_s", 60.0)))
                try:
                    st = disp.status()
                except CalibError as exc:
                    self.log({"t": time.strftime("%H:%M:%S"), "kind": "error",
                              "text": f"display unreachable: {exc}", "photo": None})
                    with self.lock:
                        self.status = "failed"
                        self.report = {"result": "needs-human",
                                       "reason": f"display unreachable: {exc}"}
                    return
                if owns_camera:
                    # The camera is initialized exactly once, here, under
                    # _camera_lock. A quick-endpoint request that slipped
                    # past its 409 guard just before start() may still
                    # hold the device — wait for it (releases within ~2 s)
                    # instead of double-opening, which Windows drivers
                    # answer with empty grabs and a bogus "busy elsewhere".
                    if not _camera_lock.acquire(timeout=_RUN_LOCK_WAIT_S):
                        raise CameraError(
                            f"camera stayed busy for {_RUN_LOCK_WAIT_S:.0f}s "
                            "(live view or check never released it)")
                    cam_lock_held = True
                    cam = Camera(int(cfg.get("camera_index", 0)),
                                 brightness=float(cfg.get("camera_brightness", 50)),
                                 exposure=_exposure_of(None, "camera_exposure", cfg),
                                 crop_percent=_crop_of(None, "camera_crop_percent", cfg))
                    cam.open()
                    try:
                        diag = cam.check_camera()
                    except CameraError as exc:
                        self.log({"t": time.strftime("%H:%M:%S"), "kind": "error",
                                  "text": f"camera check failed: {exc}", "photo": None})
                        with self.lock:
                            self.status = "failed"
                            self.report = {"result": "needs-human",
                                           "reason": f"camera check failed: {exc}"}
                        return
                    self.log({"t": time.strftime("%H:%M:%S"), "kind": "ok",
                              "text": f"camera OK: {diag}", "photo": None})
                self.log({"t": time.strftime("%H:%M:%S"), "kind": "run",
                          "text": f"started: {st.get('totalModules')} modules, "
                                  f"charset {st.get('charset')}, phase {self.phase}",
                          "photo": None})
                try:
                    snapshot = disp.snapshot()
                    with open(os.path.join(run_dir, "snapshot.json"), "w", encoding="utf-8") as fh:
                        json.dump(snapshot, fh)
                except CalibError as exc:
                    self.log({"t": time.strftime("%H:%M:%S"), "kind": "error",
                              "text": f"snapshot failed: {exc}", "photo": None})
                calib = UICalibrator(
                    disp, cam, photo_dir=run_dir,
                    dwell_ms=int(cfg.get("dwell_ms", 800)),
                    timeout_s=float(cfg.get("timeout_s", 60.0)),
                    max_phase=int(cfg.get("phase", 4)),
                    identity_thresh=float(cfg.get("identity_thresh", 0.85)),
                    relearn_templates=bool(cfg.get("relearn_templates", False)),
                    on_event=self.log,
                    abort_flag=lambda: self.abort_requested)
                with self.lock:
                    self.calib = calib
                report = calib.run()
                self.log({"t": time.strftime("%H:%M:%S"), "kind": "ok" if report["result"] == "converged" else "error",
                          "text": f"finished: {report['result']} ({report.get('reason', '')})",
                          "photo": None})
                with self.lock:
                    self.report = report
                    self.status = "aborting" if self.abort_requested else "done"
                    if self.status == "aborting":
                        self.report["reason"] = "aborted by user"
            except CameraError as exc:
                try:
                    (display or Display(cfg["display_host"])).hold(False)
                except Exception:
                    pass
                self.log({"t": time.strftime("%H:%M:%S"), "kind": "error",
                          "text": f"camera error: {exc}", "photo": None})
                with self.lock:
                    self.status = "failed"
                    self.report = {"result": "needs-human",
                                   "reason": f"camera error: {exc}"}
            except Exception as exc:  # surface crash in UI, release hold
                try:
                    (display or Display(cfg["display_host"])).hold(False)
                except Exception:
                    pass
                self.log({"t": time.strftime("%H:%M:%S"), "kind": "error",
                          "text": f"run crashed: {exc}", "photo": None})
                with self.lock:
                    self.status = "failed"
                    self.report = {"result": "needs-human", "reason": f"crash: {exc}"}
            finally:
                if owns_camera and cam is not None:
                    try:
                        cam.close()
                    except Exception:
                        pass
                if cam_lock_held:
                    _camera_lock.release()

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()
        return {"status": "running", "run_dir": run_dir}

    def abort(self):
        with self.lock:
            self.abort_requested = True
            if self.status == "running":
                self.status = "aborting"


harness = Harness()
app = FastAPI(title="Split-Flap Deterministic Calibration")


@app.get("/", response_class=HTMLResponse)
def index():
    # no-store: UI edits must never be hidden by a stale browser cache.
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as fh:
        return HTMLResponse(fh.read(), headers={"Cache-Control": "no-store"})


@app.get("/api/config")
def get_config():
    return load_config()


@app.post("/api/config")
def post_config(patch: dict):
    if patch.get("display_host") == "":
        raise HTTPException(400, "display host required")
    if "phase" in patch and patch["phase"] not in (1, 2, 3, 4, "1", "2", "3", "4", None, ""):
        raise HTTPException(400, "phase must be 1..4")
    return save_config(patch)


@app.get("/api/display")
def display_status():
    cfg = load_config()
    try:
        st = Display(cfg["display_host"]).status()
        return {"reachable": True, "totalModules": st.get("totalModules"),
                "charset": st.get("charset"),
                "contractVersion": st.get("contractVersion")}
    except CalibError as exc:
        return {"reachable": False, "error": str(exc)}


def _exposure_of(source: dict | None, key: str, cfg: dict) -> float | None:
    """Manual exposure from a request body or the saved config.

    Accepts numbers; None/""/"auto" mean auto-search. Anything else is
    a 400 — a mistyped value must not silently become auto.
    """
    raw = (source or {}).get(key, cfg.get("camera_exposure"))
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in ("", "auto"):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise HTTPException(400, f"{key} must be a number or empty (auto)")


def _crop_of(source: dict | None, key: str, cfg: dict) -> float:
    """Top/bottom crop percent from a request body or the saved config.

    Clamped to the slider range (CROP_MIN..CROP_MAX); missing or
    empty means the default. Garbage (incl. NaN/inf) is a 400 — a
    mistyped value must not silently change the framing.
    """
    raw = (source or {}).get(key, cfg.get("camera_crop_percent",
                                          DEFAULT_CROP_PERCENT))
    if raw in (None, ""):
        return DEFAULT_CROP_PERCENT
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise HTTPException(400, f"{key} must be a number 0..40")
    if not math.isfinite(value):
        raise HTTPException(400, f"{key} must be a number 0..40")
    return max(CROP_MIN, min(CROP_MAX, value))


@app.post("/api/check-camera")
def check_camera(body: dict | None = None):
    cfg = load_config()
    if harness.state()["status"] in ("running", "aborting"):
        raise HTTPException(409, "camera busy: run in progress")
    index = int((body or {}).get("camera_index", cfg.get("camera_index", 0)))
    brightness = float((body or {}).get("camera_brightness",
                                        cfg.get("camera_brightness", 50)))
    exposure = _exposure_of(body, "camera_exposure", cfg)
    crop_percent = _crop_of(body, "camera_crop_percent", cfg)
    # Brief bounded wait: an in-flight live-view poll releases the device
    # within ~2 s. Never open the camera concurrently with anyone.
    if not _camera_lock.acquire(timeout=_CHECK_LOCK_WAIT_S):
        raise HTTPException(409, "camera busy: live view or run holds it")
    cam = Camera(index, brightness=brightness, exposure=exposure,
                 crop_percent=crop_percent)
    try:
        cam.open()
        diag = cam.check_camera()
        return {"ok": True, "diagnostics": diag}
    except CameraError as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        cam.close()
        _camera_lock.release()


@app.get("/api/camera/frame")
def camera_frame(camera_index: int | None = None, brightness: float | None = None,
                 exposure: str | None = None, crop_percent: str | None = None):
    """Single live JPEG frame for the UI preview (no run needed).

    Opens the camera, grabs one frame with a short warm-up
    (Camera.open(quick=True)) and closes it again. Refused while a run
    owns the camera, or while any other request holds the device —
    never opened concurrently (Windows drivers answer that with empty
    grabs, which is how runs used to die right after start).
    """
    if harness.state()["status"] in ("running", "aborting"):
        raise HTTPException(409, "camera busy: run in progress")
    cfg = load_config()
    index = int(camera_index) if camera_index is not None else int(cfg.get("camera_index", 0))
    if brightness is None:
        brightness = float(cfg.get("camera_brightness", 50))
    if exposure is not None and exposure.strip().lower() not in ("", "auto"):
        try:
            exposure_val: float | None = float(exposure)
        except ValueError:
            raise HTTPException(400, "exposure must be a number or 'auto'")
    else:
        # Auto: let the driver's AE run (None = no manual override). The
        # AE result beats any manual value on drivers whose AE adds gain.
        exposure_val = None
    if crop_percent is not None and crop_percent.strip().lower() not in ("", "auto"):
        try:
            crop_val = float(crop_percent)
        except ValueError:
            raise HTTPException(400, "crop_percent must be a number 0..40")
        if not math.isfinite(crop_val):
            raise HTTPException(400, "crop_percent must be a number 0..40")
        crop_val = max(CROP_MIN, min(CROP_MAX, crop_val))
    else:
        crop_val = _crop_of(None, "camera_crop_percent", cfg)
    if not _camera_lock.acquire(blocking=False):
        raise HTTPException(409, "camera busy: check or run holds it")
    cam = Camera(index, brightness=float(brightness), exposure=exposure_val,
                 crop_percent=crop_val)
    try:
        cam.open(quick=True)
        frame = cam.capture()
    except CameraError as exc:
        raise HTTPException(502, f"camera error: {exc}")
    finally:
        cam.close()
        _camera_lock.release()
    h, w = frame.shape[:2]
    if w > 960:
        frame = cv2.resize(frame, (960, int(h * 960 / w)))
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        raise HTTPException(502, "JPEG encode failed")
    return Response(content=bytes(buf), media_type="image/jpeg")


@app.post("/api/run/start")
def run_start():
    cfg = load_config()
    if not cfg.get("display_host"):
        raise HTTPException(400, "configure display_host first")
    return harness.start(cfg)


@app.post("/api/run/abort")
def run_abort():
    harness.abort()
    return {"status": harness.state()["status"]}


@app.get("/api/run/state")
def run_state():
    return harness.state()


@app.get("/api/photos/{name}")
def photo(name: str):
    if "/" in name or name.startswith("."):
        raise HTTPException(400, "bad photo name")
    if not name.endswith(".png"):
        # Run dirs also hold snapshot.json (settings incl. secrets),
        # report.json and events.jsonl: never serve non-photos.
        raise HTTPException(404, "no such photo")
    with harness.lock:
        run_dir = harness.run_dir
    if not run_dir:  # no run yet: never resolve against the server CWD
        raise HTTPException(404, "no run yet")
    path = os.path.join(run_dir, name)
    if not os.path.isfile(path):
        raise HTTPException(404, "no such photo")
    return FileResponse(path, media_type="image/png")


def _templates_dir() -> str:
    with harness.lock:
        run_dir = harness.run_dir
    if not run_dir:
        raise HTTPException(404, "no run yet")
    return os.path.join(run_dir, "templates")


@app.get("/api/run/templates")
def template_list():
    with harness.lock:
        run_dir = harness.run_dir
    if not run_dir:
        return {"glyphs": [], "source": None}
    try:
        with open(os.path.join(run_dir, "templates", "manifest.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
    except OSError:
        return {"glyphs": [], "source": None}
    return {"glyphs": sorted(manifest.get("glyphs", {}).keys()),
            "source": manifest.get("source")}


@app.get("/api/run/templates/{glyph}")
def template_image(glyph: str):
    if len(glyph) != 1:
        raise HTTPException(400, "one glyph expected")
    path = os.path.join(_templates_dir(), f"glyph_U{ord(glyph):04X}.png")
    if not os.path.isfile(path):
        raise HTTPException(404, "no template for glyph")
    return FileResponse(path, media_type="image/png")


@app.post("/api/run/restore-snapshot")
def restore_snapshot():
    path = os.path.join(harness.state()["run_dir"], "snapshot.json")
    try:
        with open(path, encoding="utf-8") as fh:
            snapshot = json.load(fh)
    except OSError:
        raise HTTPException(404, "no snapshot from a run yet")
    cfg = load_config()
    try:
        return Display(cfg["display_host"]).restore(snapshot)
    except CalibError as exc:
        raise HTTPException(502, str(exc))


def main():
    import uvicorn

    uvicorn.run(app, host=os.environ.get("CALIB_HOST", "127.0.0.1"),
                port=int(os.environ.get("CALIB_PORT", "8001")),
                access_log=False)  # state poll every 1.5s would spam the log


if __name__ == "__main__":
    main()
