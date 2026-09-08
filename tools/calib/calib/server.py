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
import os
import threading
import time

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response

from .camera import Camera, CameraError
from .display import CalibError, Display
from .loop import Calibrator

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


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


DEFAULTS = {
    "display_host": "splitflap.local",
    "camera_index": 0,
    "camera_brightness": 50,
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
            stored[key] = patch[key]
    with open(config_path(), "w", encoding="utf-8") as fh:
        json.dump(stored, fh, indent=2)
    os.chmod(config_path(), 0o600)
    return load_config()


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
        rec = super().shoot(frame, tag)
        scores = self.scores(rec)
        bad = [(i, s["verdict"]) for i, s in enumerate(scores)
               if s["verdict"] != "ok"]
        summary = f"{len(scores) - len(bad)}/{len(scores)} ok" if scores else "no crops"
        if bad:
            summary += f" — off: {', '.join(f'm{i} {v}' for i, v in bad)}"
        self.event("photo", f"{tag} frameId={rec['frameId']} "
                            f"show={frame!r} → {summary}",
                   photo=os.path.basename(rec["photo"]),
                   detail={"frame": frame, "frameId": rec["frameId"], "tag": tag,
                           "scores": [{"module": i, **s}
                                      for i, s in enumerate(scores)]})
        return rec

    def _tune_cell(self, module: int, char_index: int, show_frame: str) -> dict:
        self.event("tune", f"tuning module {module} "
                           f"{'coarse' if char_index < 0 else f'char {char_index}'} …")
        out = super()._tune_cell(module, char_index, show_frame)
        last = self.deltas[-1] if self.deltas else {}
        if last.get("proposal"):
            self.event("tune", f"module {module}: proposal only "
                               f"(phase 1/2, hardware untouched)")
        elif last.get("new") != last.get("old"):
            self.event("tune", f"module {module}: {last.get('old')} → "
                               f"{last.get('new')} (kept)")
        else:
            self.event("tune", f"module {module}: no improvement, kept "
                               f"{last.get('old')}")
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

    def log(self, event: dict):
        with self.lock:
            self.events.append(event)
            if event.get("photo") and event["photo"] not in self.photos:
                self.photos.append(event["photo"])
            self.events = self.events[-500:]

    def state(self) -> dict:
        with self.lock:
            calib = self.calib
            return {"status": self.status, "events": self.events[-100:],
                    "photos": self.photos[-24:], "report": self.report,
                    "run_dir": self.run_dir, "phase": self.phase,
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
            self.abort_requested = False
        runs_dir = os.path.join(data_dir(), "runs")
        run_dir = alloc_run_dir(runs_dir)
        with self.lock:
            self.run_dir = run_dir

        def _run():
            cam = camera
            owns_camera = cam is None
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
                    cam = Camera(int(cfg.get("camera_index", 0)),
                                 brightness=float(cfg.get("camera_brightness", 50)))
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
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as fh:
        return fh.read()


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


@app.post("/api/check-camera")
def check_camera(body: dict | None = None):
    cfg = load_config()
    index = int((body or {}).get("camera_index", cfg.get("camera_index", 0)))
    brightness = float((body or {}).get("camera_brightness",
                                        cfg.get("camera_brightness", 50)))
    cam = Camera(index, brightness=brightness)
    try:
        cam.open()
        diag = cam.check_camera()
        return {"ok": True, "diagnostics": diag}
    except CameraError as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        cam.close()


@app.get("/api/camera/frame")
def camera_frame(camera_index: int | None = None, brightness: float | None = None):
    """Single live JPEG frame for the UI preview (no run needed).

    Opens the camera, grabs one frame with a short warm-up
    (Camera.open(quick=True)) and closes it again. Refused while a run
    owns the camera.
    """
    if harness.state()["status"] in ("running", "aborting"):
        raise HTTPException(409, "camera busy: run in progress")
    cfg = load_config()
    index = int(camera_index) if camera_index is not None else int(cfg.get("camera_index", 0))
    if brightness is None:
        brightness = float(cfg.get("camera_brightness", 50))
    cam = Camera(index, brightness=float(brightness))
    try:
        cam.open(quick=True)
        frame = cam.capture()
    except CameraError as exc:
        raise HTTPException(502, f"camera error: {exc}")
    finally:
        cam.close()
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
                port=int(os.environ.get("CALIB_PORT", "8001")))


if __name__ == "__main__":
    main()
