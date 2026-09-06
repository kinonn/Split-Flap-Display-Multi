"""VLM calibration harness web app.

Configure once (display host + LLM provider/model/key), then drive runs
from the browser. The API key lives server-side only (env or data dir).
"""

from __future__ import annotations

import json
import os
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from urllib.parse import urlparse

from calib.camera import Camera, CameraError
from calib.display import CalibError, Display
from calib.loop import Calibrator

from .agent import Agent, load_system_prompt
from .vlm import VLMClient

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def data_dir() -> str:
    return os.environ.get("CALIB_AGENT_DATA", os.path.join(os.getcwd(), "data"))


def config_path() -> str:
    return os.path.join(data_dir(), "config.json")


def load_config() -> dict:
    cfg: dict = {}
    try:
        with open(config_path()) as fh:
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
    cfg.setdefault("mode", "full")
    return cfg


def save_config(patch: dict) -> dict:
    os.makedirs(data_dir(), exist_ok=True)
    cfg = load_config()
    # Never clobber the file-backed key with env-provided values on read;
    # here we only store what the UI sent (empty key keeps the old one).
    stored = {}
    try:
        with open(config_path()) as fh:
            stored = json.load(fh)
    except (OSError, ValueError):
        pass
    for key in ("display_host", "llm_base_url", "llm_model", "camera_index",
                "identity_thresh", "phase"):
        if key in patch and patch[key] not in (None, ""):
            stored[key] = patch[key]
    if patch.get("llm_api_key"):
        stored["llm_api_key"] = patch["llm_api_key"]
    if "mode" in patch:
        if patch["mode"] not in ("dry-run", "preview", "full"):
            raise HTTPException(400, "mode must be dry-run, preview or full")
        stored["mode"] = patch["mode"]
    with open(config_path(), "w") as fh:
        json.dump(stored, fh, indent=2)
    os.chmod(config_path(), 0o600)
    return masked_config()


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

    def log(self, event: dict):
        with self.lock:
            self.events.append(event)
            if event.get("photo") and event["photo"] not in self.photos:
                self.photos.append(event["photo"])
            self.events = self.events[-500:]

    def state(self) -> dict:
        with self.lock:
            return {"status": self.status, "events": self.events[-100:],
                    "photos": self.photos[-24:], "report": self.report,
                    "run_dir": self.run_dir, "mode": self.mode,
                    "steps": self.agent.steps if self.agent else 0}

    def start(self, cfg: dict) -> dict:
        with self.lock:
            if self.status == "running":
                raise HTTPException(409, "run already in progress")
            self.status = "running"
            self.mode = cfg.get("mode", "full")
            self.events = []
            self.photos = []
            self.report = None
        runs_dir = os.path.join(data_dir(), "runs")
        os.makedirs(runs_dir, exist_ok=True)
        run_dir = os.path.join(runs_dir, f"run-{len(os.listdir(runs_dir)) + 1:03d}")
        os.makedirs(run_dir, exist_ok=True)
        with self.lock:
            self.run_dir = run_dir

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
                camera = Camera(int(cfg.get("camera_index", 0)))
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
                vlm = VLMClient(cfg["llm_base_url"], cfg["llm_model"], cfg["llm_api_key"])
                calib = Calibrator(display, camera, photo_dir=run_dir,
                                   identity_thresh=float(cfg.get("identity_thresh", 0.85)))
                agent = Agent(vlm, calib, load_system_prompt(), on_event=self.log,
                              mode=cfg.get("mode", "full"))
                with self.lock:
                    self.agent = agent
                self.report = agent.run()
                with self.lock:
                    self.report = agent.report
                    self.status = "done"
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
    with open(os.path.join(STATIC_DIR, "index.html")) as fh:
        return fh.read()


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


@app.post("/api/check-camera")
def check_camera(body: dict | None = None):
    cfg = load_config()
    index = int((body or {}).get("camera_index", cfg.get("camera_index", 0)))
    cam = Camera(index)
    try:
        cam.open()
        diag = cam.check_camera()
        return {"ok": True, "diagnostics": diag}
    except CameraError as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        cam.close()


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
        with open(os.path.join(run_dir, "templates", "manifest.json")) as fh:
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
        with open(path) as fh:
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
                port=int(os.environ.get("CALIB_AGENT_PORT", "8000")))


if __name__ == "__main__":
    main()
