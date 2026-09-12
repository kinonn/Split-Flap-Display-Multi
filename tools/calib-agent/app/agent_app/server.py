"""VLM calibration harness web app.

Configure once (display host + LLM provider/model/key), then drive runs
from the browser. The API key lives server-side only (env or data dir).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import threading
import uuid

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from urllib.parse import urlparse

from calib.camera import (CROP_MAX, CROP_MIN, DEFAULT_CROP_PERCENT, Camera,
                          CameraError)
from calib.display import CalibError, Display
from calib.loop import Calibrator

from .agent import FULL_DRUM_PROMPT_EXTRA, Agent, load_system_prompt
from .vlm import VLMClient

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Camera ownership: the run is the ONLY component that opens the camera
# for real work, and it does so under this lock for the whole run. The
# quick endpoints (check, live frame) take it non-blocking/bounded and
# answer 409 when busy — FastAPI runs each request in its own thread and
# Windows webcam drivers handle concurrent opens badly (empty grabs).
_camera_lock = threading.Lock()
# Bounded waits so nobody ever opens the device twice: a live-view poll
# releases it within ~2 s, so the check briefly waits for stragglers and
# the run waits a little longer before giving up.
_CHECK_LOCK_WAIT_S = 2.5
_RUN_LOCK_WAIT_S = 30.0


def data_dir() -> str:
    return os.environ.get("CALIB_AGENT_DATA", os.path.join(os.getcwd(), "data"))


def alloc_run_dir(runs_dir: str) -> str:
    """Allocate the first unused run-NNN dir (collision-proof, see calib UI)."""
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
    """Stable OpenCode session id, derived from the API key.

    The VLM system prompt is runbook-sized, and the provider caches the
    prompt prefix per session. A fresh random id per run (the old
    behavior) made the FIRST request of every run a full cold ingest of
    the system prompt — visible as a ~45 s pause between "run started"
    and the first show. Deriving the id from the key keeps the cache
    warm from turn 1 of every run; a different key/url gets a different
    session automatically.
    """
    scope = str(cfg.get("llm_api_key") or cfg.get("llm_base_url") or "local")
    return uuid.uuid5(uuid.NAMESPACE_URL, "splitflap-calib:" + scope).hex


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
    cfg.setdefault("camera_exposure", None)  # None = driver AE; a value fixes the sensor
    cfg.setdefault("camera_crop_percent", DEFAULT_CROP_PERCENT)  # % trimmed top AND bottom (0..40)
    cfg.setdefault("mode", "full")
    cfg.setdefault("full_drum", False)
    return cfg


def save_config(patch: dict) -> dict:
    os.makedirs(data_dir(), exist_ok=True)
    cfg = load_config()
    # Never clobber the file-backed key with env-provided values on read;
    # here we only store what the UI sent (empty key keeps the old one).
    stored = {}
    try:
        with open(config_path(), encoding="utf-8") as fh:
            stored = json.load(fh)
    except (OSError, ValueError):
        pass
    for key in ("display_host", "llm_base_url", "llm_model", "camera_index",
                "camera_brightness", "identity_thresh", "phase"):
        if key in patch and patch[key] not in (None, ""):
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
    if patch.get("llm_api_key"):
        stored["llm_api_key"] = patch["llm_api_key"]
    if "mode" in patch:
        if patch["mode"] not in ("dry-run", "preview", "full"):
            raise HTTPException(400, "mode must be dry-run, preview or full")
        stored["mode"] = patch["mode"]
    # Boolean toggle: accept real booleans (and 0/1 from form posts).
    if "full_drum" in patch:
        stored["full_drum"] = bool(patch["full_drum"])
    _atomic_write_json(config_path(), stored)
    return masked_config()


def _atomic_write_json(path: str, payload: dict) -> None:
    """Write JSON with mode 0600, atomically (issue kinonn-bot#36).

    The old write-then-chmod left the API-key file world-readable when a
    crash landed between the two syscalls, and a mid-write crash left a
    truncated config. Writing to a same-dir temp file, chmodding it, then
    os.replace() closes both windows. The temp name carries pid + thread
    id: the sync /api/config endpoints run on a threadpool, so two
    concurrent saves must not share one temp file.
    """
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
        self.agent: Agent | None = None
        self.thread: threading.Thread | None = None
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
            return {"status": self.status, "events": self.events[-100:],
                    "photos": self.photos[-24:], "report": self.report,
                    "run_dir": self.run_dir, "mode": self.mode,
                    "run_seq": self.run_seq,
                    "steps": self.agent.steps if self.agent else 0}

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
            try:
                try:
                    display = Display(cfg["display_host"])
                    display.status()
                except CalibError as exc:
                    self.log({"t": "", "kind": "error",
                              "text": f"display unreachable: {exc}", "photo": None})
                    with self.lock:
                        self.status = "failed"
                        self.report = {"result": "needs-human",
                                       "reason": f"display unreachable: {exc}"}
                    return
                # The camera is initialized exactly once, here, under
                # _camera_lock. A quick-endpoint request that slipped
                # past its 409 guard just before start() may still hold
                # the device — wait for it (releases within ~2 s)
                # instead of double-opening, which Windows drivers
                # answer with empty grabs and a bogus "busy elsewhere".
                if not _camera_lock.acquire(timeout=_RUN_LOCK_WAIT_S):
                    raise CameraError(
                        f"camera stayed busy for {_RUN_LOCK_WAIT_S:.0f}s "
                        "(live view or check never released it)")
                try:
                    camera = Camera(int(cfg.get("camera_index", 0)),
                                    brightness=float(cfg.get("camera_brightness", 50)),
                                    exposure=_exposure_of(None, "camera_exposure", cfg),
                                    crop_percent=_crop_of(None, "camera_crop_percent", cfg))
                    camera.open()
                    try:
                        camera.check_camera()
                    except CameraError as exc:
                        self.log({"t": "", "kind": "error",
                                  "text": f"camera check failed: {exc}", "photo": None})
                        with self.lock:
                            self.status = "failed"
                            self.report = {"result": "needs-human",
                                           "reason": f"camera check failed: {exc}"}
                        return
                    vlm = VLMClient(cfg["llm_base_url"], cfg["llm_model"],
                                    cfg["llm_api_key"],
                                    session_id=vlm_session_id(cfg))
                    calib = Calibrator(display, camera, photo_dir=run_dir,
                                       identity_thresh=float(cfg.get("identity_thresh", 0.85)),
                                       full=bool(cfg.get("full_drum", False)))
                    prompt = load_system_prompt()
                    if cfg.get("full_drum"):
                        prompt += "\n\n" + FULL_DRUM_PROMPT_EXTRA
                    agent = Agent(vlm, calib, prompt, on_event=self.log,
                                  mode=cfg.get("mode", "full"))
                    with self.lock:
                        self.agent = agent
                    self.report = agent.run()
                    with self.lock:
                        self.report = agent.report
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
                self.log({"t": "", "kind": "error", "text": f"run crashed: {exc}",
                          "photo": None})
                with self.lock:
                    self.status = "failed"
                    self.report = {"result": "needs-human", "reason": f"crash: {exc}"}
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
            if self.agent:
                self.agent.aborted = True
            if self.status == "running":
                self.status = "aborting"


harness = Harness()
app = FastAPI(title="Split-Flap VLM Calibration Harness")


@app.get("/", response_class=HTMLResponse)
def index():
    # no-store: UI edits must never be hidden by a stale browser cache.
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
    for key in ("display_host", "llm_base_url", "llm_model"):
        if not cfg.get(key):
            raise HTTPException(400, f"configure {key} first")
    host = (urlparse(str(cfg["llm_base_url"])).hostname or "").lower()
    if not cfg.get("llm_api_key") and host not in ("localhost", "127.0.0.1",
                                                  "0.0.0.0", "::1"):
        raise HTTPException(400, "configure llm_api_key first "
                                 "(not needed for local base URLs)")
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

    uvicorn.run(app, host=os.environ.get("CALIB_AGENT_HOST", "127.0.0.1"),
                port=int(os.environ.get("CALIB_AGENT_PORT", "8000")),
                access_log=False)  # state poll every 1.5s would spam the log


if __name__ == "__main__":
    main()
