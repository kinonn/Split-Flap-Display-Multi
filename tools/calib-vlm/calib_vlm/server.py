"""VLM-reader calibration web app.

Configure once (display host + reader provider/model/key + camera), then
drive runs from the browser. The API key lives server-side only (env or
data dir, mode 0600).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import threading
import time
import uuid

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from urllib.parse import urlparse

from calib.camera import (CROP_MAX, CROP_MIN, DEFAULT_CROP_PERCENT, Camera,
                          CameraError)
from calib.display import CalibError, Display

from .calibrate import VlmCalibrator
from .reader import ReaderError, VlmReader, annotate_modules, jpeg_bytes
from .vlm import VLMClient

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Camera ownership: the run is the ONLY component that opens the camera
# for real work, and it does so under this lock for the whole run. Quick
# endpoints (check, live frame, read test) take it non-blocking/bounded
# and answer 409 when busy — Windows webcam drivers handle concurrent
# opens badly (empty grabs).
_camera_lock = threading.Lock()
_CHECK_LOCK_WAIT_S = 2.5
_RUN_LOCK_WAIT_S = 30.0


def data_dir() -> str:
    return os.environ.get("CALIB_VLM_DATA", os.path.join(os.getcwd(), "data"))


def alloc_run_dir(runs_dir: str) -> str:
    """Allocate the first unused run-NNN dir (collision-proof)."""
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


def vlm_session_id(cfg: dict) -> str:
    """Stable OpenCode session id derived from the API key (prompt cache)."""
    scope = str(cfg.get("llm_api_key") or cfg.get("llm_base_url") or "local")
    return uuid.uuid5(uuid.NAMESPACE_URL, "splitflap-calib-vlm:" + scope).hex


def clear_previous_runs(runs_dir: str) -> int:
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


def load_config() -> dict:
    cfg: dict = {}
    try:
        with open(config_path(), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        pass
    env_map = {"display_host": "DISPLAY_HOST", "llm_base_url": "LLM_BASE_URL",
               "llm_model": "LLM_MODEL", "llm_api_key": "LLM_API_KEY",
               "camera_index": "CAMERA_INDEX"}
    for key, env in env_map.items():
        if env in os.environ and os.environ[env]:
            cfg[key] = os.environ[env]
    cfg.setdefault("display_host", "splitflap.local")
    cfg.setdefault("llm_base_url", "https://opencode.ai/zen/go/v1")
    cfg.setdefault("llm_model", "deepseek-v4-flash-vision-exp")
    cfg.setdefault("camera_index", 0)
    cfg.setdefault("camera_brightness", 50)
    cfg.setdefault("camera_exposure", None)
    cfg.setdefault("camera_crop_percent", DEFAULT_CROP_PERCENT)
    cfg.setdefault("camera_warmup_s", 5.0)
    cfg.setdefault("mode", "full")
    if cfg.get("mode") == "preview":  # legacy mode, folded into full
        cfg["mode"] = "full"
    cfg.setdefault("exhaustive", False)
    cfg.setdefault("annotate", True)
    cfg.setdefault("dwell_ms", 800)
    cfg.setdefault("timeout_s", 60.0)
    cfg.setdefault("min_confidence", 0.6)
    cfg.setdefault("max_seconds", 5400.0)
    return cfg


def save_config(patch: dict) -> dict:
    os.makedirs(data_dir(), exist_ok=True)
    stored = {}
    try:
        with open(config_path(), encoding="utf-8") as fh:
            stored = json.load(fh)
    except (OSError, ValueError):
        pass
    for key in ("display_host", "llm_base_url", "llm_model", "camera_index",
                "camera_brightness", "mode", "exhaustive", "annotate",
                "dwell_ms", "timeout_s", "min_confidence", "max_seconds"):
        if key in patch and patch[key] not in (None, ""):
            stored[key] = patch[key]
    for key, lo, hi in (("dwell_ms", 0, 10000), ("timeout_s", 1, 3600),
                        ("min_confidence", 0, 1), ("max_seconds", 60, 36000)):
        if key in patch and patch[key] not in (None, ""):
            try:
                value = float(patch[key])
            except (TypeError, ValueError):
                raise HTTPException(400, f"{key} must be a number")
            if not math.isfinite(value):
                raise HTTPException(400, f"{key} must be a number")
            stored[key] = int(value) if key in ("dwell_ms", "max_seconds") \
                else value
    # Start-wait (warm-up budget for cameras that open black) validates
    # and clamps to 0..30 s.
    if "camera_warmup_s" in patch and patch["camera_warmup_s"] not in (None, ""):
        try:
            value = float(patch["camera_warmup_s"])
        except (TypeError, ValueError):
            raise HTTPException(400, "camera_warmup_s must be a number 0..30")
        if not math.isfinite(value):
            raise HTTPException(400, "camera_warmup_s must be a number 0..30")
        stored["camera_warmup_s"] = max(0.0, min(30.0, value))
    if "mode" in patch and patch["mode"] not in ("dry-run", "full"):
        raise HTTPException(400, "mode must be dry-run or full")
    # Exposure is settable AND clearable (null/"" = back to auto-search).
    if "camera_exposure" in patch:
        raw = patch["camera_exposure"]
        if raw in (None, "") or (isinstance(raw, str)
                                 and raw.strip().lower() == "auto"):
            stored["camera_exposure"] = None
        else:
            try:
                stored["camera_exposure"] = float(raw)
            except (TypeError, ValueError):
                raise HTTPException(400, "camera_exposure must be a number "
                                         "or empty (auto)")
    # Crop is validated + clamped to the slider range (0..40, default 30).
    if "camera_crop_percent" in patch:
        raw = patch["camera_crop_percent"]
        if raw in (None, ""):
            stored["camera_crop_percent"] = DEFAULT_CROP_PERCENT
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                raise HTTPException(400, "camera_crop_percent must be a "
                                         "number 0..40")
            if not math.isfinite(value):
                raise HTTPException(400, "camera_crop_percent must be a "
                                         "number 0..40")
            stored["camera_crop_percent"] = max(CROP_MIN, min(CROP_MAX, value))
    if patch.get("llm_api_key"):
        stored["llm_api_key"] = patch["llm_api_key"]
    if not stored.get("mode"):
        stored["mode"] = "full"
    _atomic_write_json(config_path(), stored)
    return masked_config()


def _atomic_write_json(path: str, payload: dict) -> None:
    """Write JSON with mode 0600, atomically (temp file + os.replace)."""
    tmp = path + f".tmp-{os.getpid()}-{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def masked_config() -> dict:
    cfg = load_config()
    out = dict(cfg)
    if out.get("llm_api_key"):
        out["llm_api_key"] = "***" + str(out["llm_api_key"])[-4:]
    else:
        out["llm_api_key"] = ""
    return out


def _exposure_of(source: dict | None, key: str, cfg: dict) -> float | None:
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


def _warmup_of(source: dict | None, key: str, cfg: dict) -> float:
    """Warm-up budget override (start-wait slider), default 5 s."""
    raw = (source or {}).get(key, cfg.get("camera_warmup_s", 5.0))
    if raw in (None, ""):
        return 5.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise HTTPException(400, f"{key} must be a number 0..30")
    if not math.isfinite(value):
        raise HTTPException(400, f"{key} must be a number 0..30")
    return max(0.0, min(30.0, value))


class Harness:
    """Owns one run at a time (background thread) plus shared state."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.photos: list[str] = []
        self.status = "idle"
        self.mode = "full"
        self.report: dict | None = None
        self.run_dir = ""
        self.calibrator: VlmCalibrator | None = None
        self.thread: threading.Thread | None = None
        self.run_seq = 0
        self.last_readtest_photo = ""

    def log(self, event: dict):
        with self.lock:
            self.events.append(event)
            if event.get("photo") and event["photo"] not in self.photos:
                self.photos.append(event["photo"])
            # No in-memory cap: the full log stays available via
            # /api/run/events (offset/limit). events.jsonl on disk is
            # already uncapped; the UI polls pages instead of a slice.
            run_dir = self.run_dir
        if run_dir:
            try:
                with open(os.path.join(run_dir, "events.jsonl"), "a",
                          encoding="utf-8") as fh:
                    fh.write(json.dumps(event, default=str) + "\n")
            except OSError:
                pass  # logging must never break a run

    def state(self) -> dict:
        with self.lock:
            return {"status": self.status, "events": self.events[-200:],
                    "photos": self.photos[-24:], "report": self.report,
                    "run_dir": self.run_dir, "mode": self.mode,
                    "run_seq": self.run_seq,
                    "event_count": len(self.events),
                    "frames": self.calibrator.frames_used
                    if self.calibrator else 0,
                    "vlm_calls": self.calibrator.vlm_calls
                    if self.calibrator else 0}

    def events_since(self, offset: int, limit: int = 500) -> dict:
        with self.lock:
            total = len(self.events)
            offset = max(0, min(offset, total))
            limit = max(1, min(limit, 2000))
            return {"total": total, "offset": offset,
                    "events": self.events[offset:offset + limit]}

    def start(self, cfg: dict) -> dict:
        with self.lock:
            if self.status == "running":
                raise HTTPException(409, "run already in progress")
            self.status = "running"
            self.mode = cfg.get("mode", "full")
            self.events = []
            self.photos = []
            self.run_seq += 1
            self.report = None
        runs_dir = os.path.join(data_dir(), "runs")
        cleared = clear_previous_runs(runs_dir)
        run_dir = alloc_run_dir(runs_dir)
        with self.lock:
            self.run_dir = run_dir
        if cleared:
            self.log({"t": "", "kind": "run",
                      "text": f"cleared {cleared} previous run(s) "
                              "(old logs, photos, reports)", "photo": None})

        def _run():
            camera: Camera | None = None
            display: Display | None = None
            try:
                try:
                    display = Display(cfg["display_host"])
                    display.status()
                except CalibError as exc:
                    self.log({"t": "", "kind": "error",
                              "text": f"display unreachable: {exc}",
                              "photo": None})
                    with self.lock:
                        self.status = "failed"
                        self.report = {"result": "needs-human",
                                       "reason": f"display unreachable: {exc}"}
                    return
                if not _camera_lock.acquire(timeout=_RUN_LOCK_WAIT_S):
                    raise CameraError(
                        f"camera stayed busy for {_RUN_LOCK_WAIT_S:.0f}s "
                        "(live view or check never released it)")
                try:
                    camera = Camera(
                        int(cfg.get("camera_index", 0)),
                        brightness=float(cfg.get("camera_brightness", 50)),
                        exposure=_exposure_of(None, "camera_exposure", cfg),
                        crop_percent=_crop_of(None, "camera_crop_percent", cfg),
                        warmup_s=_warmup_of(None, "camera_warmup_s", cfg))
                    camera.open()
                    try:
                        camera.check_camera()
                    except CameraError as exc:
                        self.log({"t": "", "kind": "error",
                                  "text": f"camera check failed: {exc}",
                                  "photo": None})
                        with self.lock:
                            self.status = "failed"
                            self.report = {"result": "needs-human",
                                           "reason":
                                               f"camera check failed: {exc}"}
                        return
                    vlm = VLMClient(
                        cfg["llm_base_url"], cfg["llm_model"],
                        cfg["llm_api_key"], session_id=vlm_session_id(cfg))
                    reader = VlmReader(vlm,
                                       annotate=bool(cfg.get("annotate", True)))
                    calib = VlmCalibrator(
                        display, camera, reader, photo_dir=run_dir,
                        dwell_ms=int(cfg.get("dwell_ms", 800)),
                        timeout_s=float(cfg.get("timeout_s", 60.0)),
                        min_confidence=float(cfg.get("min_confidence", 0.6)),
                        exhaustive=bool(cfg.get("exhaustive", False)),
                        mode=cfg.get("mode", "full"), on_event=self.log,
                        max_seconds=float(cfg.get("max_seconds", 5400.0)),
                        run_context={
                            "display_host": cfg.get("display_host"),
                            "llm_base_url": cfg.get("llm_base_url"),
                            "llm_model": cfg.get("llm_model"),
                            "camera_index": cfg.get("camera_index"),
                            "camera_brightness": cfg.get("camera_brightness"),
                            "camera_exposure": cfg.get("camera_exposure"),
                            "camera_crop_percent": cfg.get(
                                "camera_crop_percent"),
                            "camera_warmup_s": cfg.get("camera_warmup_s"),
                            "annotate": bool(cfg.get("annotate", True)),
                        })
                    with self.lock:
                        self.calibrator = calib
                    self.report = calib.run()
                    with self.lock:
                        self.report = calib.report
                        self.status = "done"
                finally:
                    _camera_lock.release()
            except CameraError as exc:
                try:
                    Display(cfg["display_host"]).hold(False)
                except Exception:
                    pass
                self.log({"t": "", "kind": "error",
                          "text": f"camera error: {exc}", "photo": None})
                with self.lock:
                    self.status = "failed"
                    self.report = {"result": "needs-human",
                                   "reason": f"camera error: {exc}"}
            except Exception as exc:  # surface crash in UI, release hold
                try:
                    Display(cfg["display_host"]).hold(False)
                except Exception:
                    pass
                self.log({"t": "", "kind": "error",
                          "text": f"run crashed: {exc}", "photo": None})
                with self.lock:
                    self.status = "failed"
                    self.report = {"result": "needs-human",
                                   "reason": f"crash: {exc}"}
            finally:
                if camera is not None:
                    try:
                        camera.close()
                    except Exception:
                        pass

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()
        return {"status": "running", "run_dir": run_dir}

    def abort(self):
        with self.lock:
            if self.calibrator:
                self.calibrator.abort()
            if self.status == "running":
                self.status = "aborting"


harness = Harness()
app = FastAPI(title="Split-Flap VLM Calibration")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as fh:
        return HTMLResponse(fh.read(), headers={"Cache-Control": "no-store"})


@app.get("/api/config")
def get_config():
    return masked_config()


@app.post("/api/config")
def post_config(patch: dict):
    if patch.get("display_host") == "":
        raise HTTPException(400, "display host required")
    return save_config(patch)


@app.get("/api/display")
def display_status():
    cfg = load_config()
    try:
        st = Display(cfg["display_host"]).status()
        return {"reachable": True, "totalModules": st.get("totalModules"),
                "numModules": st.get("numModules"),
                "charset": st.get("charset"),
                "drumOrder": st.get("drumOrder"),
                "contractVersion": st.get("contractVersion")}
    except CalibError as exc:
        return {"reachable": False, "error": str(exc)}


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
    warmup_s = _warmup_of(body, "camera_warmup_s", cfg)
    if not _camera_lock.acquire(timeout=_CHECK_LOCK_WAIT_S):
        raise HTTPException(409, "camera busy: live view or run holds it")
    cam = Camera(index, brightness=brightness, exposure=exposure,
                 crop_percent=crop_percent, warmup_s=warmup_s)
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
def camera_frame(camera_index: int | None = None,
                 brightness: float | None = None,
                 exposure: str | None = None,
                 crop_percent: str | None = None):
    """Single live JPEG frame for the UI preview (no run needed)."""
    if harness.state()["status"] in ("running", "aborting"):
        raise HTTPException(409, "camera busy: run in progress")
    cfg = load_config()
    index = int(camera_index) if camera_index is not None else int(
        cfg.get("camera_index", 0))
    if brightness is None:
        brightness = float(cfg.get("camera_brightness", 50))
    if exposure is not None and exposure.strip().lower() not in ("", "auto"):
        try:
            exposure_val: float | None = float(exposure)
        except ValueError:
            raise HTTPException(400, "exposure must be a number or 'auto'")
    else:
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


READ_TEST_KEEP = 20


def _prune_read_tests(out_dir: str) -> None:
    """Keep only the newest read-test photos (the folder grows per click)."""
    try:
        files = sorted(os.path.join(out_dir, name)
                       for name in os.listdir(out_dir)
                       if name.endswith(".png"))
        for old in files[:-READ_TEST_KEEP]:
            os.remove(old)
    except OSError:
        pass


def _reader_from_config(cfg: dict) -> VlmReader:
    if not cfg.get("llm_base_url") or not cfg.get("llm_model"):
        raise HTTPException(400, "configure llm_base_url and llm_model first")
    host = (urlparse(str(cfg["llm_base_url"])).hostname or "").lower()
    if not cfg.get("llm_api_key") and host not in ("localhost", "127.0.0.1",
                                                  "0.0.0.0", "::1"):
        raise HTTPException(400, "configure llm_api_key first "
                                 "(not needed for local base URLs)")
    vlm = VLMClient(cfg["llm_base_url"], cfg["llm_model"],
                    cfg.get("llm_api_key", ""),
                    session_id=vlm_session_id(cfg))
    return VlmReader(vlm, annotate=bool(cfg.get("annotate", True)))


@app.post("/api/read-test")
def read_test(body: dict | None = None):
    """Show one pattern and return the reader's per-module transcription."""
    if harness.state()["status"] in ("running", "aborting"):
        raise HTTPException(409, "run in progress")
    cfg = load_config()
    reader = _reader_from_config(cfg)
    wanted = str((body or {}).get("frame", ""))
    if not _camera_lock.acquire(timeout=_CHECK_LOCK_WAIT_S):
        raise HTTPException(409, "camera busy: live view or run holds it")
    cam = Camera(int(cfg.get("camera_index", 0)),
                 brightness=float(cfg.get("camera_brightness", 50)),
                 exposure=_exposure_of(body, "camera_exposure", cfg),
                 crop_percent=_crop_of(body, "camera_crop_percent", cfg),
                 warmup_s=_warmup_of(body, "camera_warmup_s", cfg))
    display = Display(cfg["display_host"])
    engaged = False
    try:
        status = display.status()
        total = int(status["totalModules"])
        drum = str(status["drumOrder"])
        if len(wanted) > total:
            raise HTTPException(400, f"frame {len(wanted)} chars is longer "
                                     f"than the {total}-module display")
        frame = wanted.ljust(total)
        if not status.get("holdActive"):
            display.hold(True)
            engaged = True
        display.show_and_settle(frame, int(cfg.get("dwell_ms", 800)),
                                float(cfg.get("timeout_s", 60.0)))
        cam.open()
        img = cam.capture()
    except (CalibError, CameraError) as exc:
        raise HTTPException(502, str(exc))
    finally:
        if engaged:
            try:
                display.hold(False)
            except Exception:
                pass
        cam.close()
        _camera_lock.release()
    send = annotate_modules(img, total) if reader.annotate else img
    try:
        reading = reader.read(jpeg_bytes(send), total=total, expected=frame,
                              charset=drum, drum=drum)
    except ReaderError as exc:
        raise HTTPException(502, f"reader failed: {exc}")
    out_dir = os.path.join(data_dir(), "read-tests")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"readtest_{int(time.time())}.png")
    cv2.imwrite(path, img)
    _prune_read_tests(out_dir)
    with harness.lock:
        harness.last_readtest_photo = path
    detail = {"frame": frame, **reading.as_dict()}
    return {"ok": True, "photo": "/api/read-test/photo", **detail}


@app.get("/api/read-test/photo")
def read_test_photo():
    with harness.lock:
        path = harness.last_readtest_photo
    if not path or not os.path.isfile(path):
        raise HTTPException(404, "no read-test photo yet")
    return FileResponse(path, media_type="image/png")


@app.post("/api/run/start")
def run_start():
    cfg = load_config()
    for key in ("display_host", "llm_base_url", "llm_model"):
        if not cfg.get(key):
            raise HTTPException(400, f"configure {key} first")
    _reader_from_config(cfg)  # validates provider/key like the read test
    return harness.start(cfg)


@app.post("/api/run/abort")
def run_abort():
    harness.abort()
    return {"status": harness.state()["status"]}


@app.get("/api/run/state")
def run_state():
    return harness.state()


@app.get("/api/run/events")
def run_events(offset: int = 0, limit: int = 500):
    """Full log paging: the UI polls this so no event is ever dropped."""
    return harness.events_since(offset, limit)


@app.get("/api/photos/{name}")
def photo(name: str):
    if "/" in name or name.startswith("."):
        raise HTTPException(400, "bad photo name")
    if not name.endswith(".png"):
        raise HTTPException(404, "no such photo")
    with harness.lock:
        run_dir = harness.run_dir
    if not run_dir:
        raise HTTPException(404, "no run yet")
    path = os.path.join(run_dir, name)
    if not os.path.isfile(path):
        raise HTTPException(404, "no such photo")
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

    uvicorn.run(app, host=os.environ.get("CALIB_VLM_HOST", "127.0.0.1"),
                port=int(os.environ.get("CALIB_VLM_PORT", "8002")),
                access_log=False)


if __name__ == "__main__":
    main()
