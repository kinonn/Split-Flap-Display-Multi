"""Unified calib-auto web server.

One FastAPI app (default port 8004) serving all four surfaces of the
tool:

- **Calibration** (``/cnn``, ``/vlm``) — run the auto-calibration loop
  with either recognizer; tailored pages per approach.
- **Golden set** (``/golden``) — curate the labeled ground truth that
  trains the CNN and scores the benchmark.
- **Train** (``/train``) — build glyph caches and train/retrain the
  CNN (or template bank) with live progress and metrics.
- **Benchmark** (``/bench``) — run any configured VLM against the
  golden set to pick the best model for the VLM approach.

Three background jobs share the same event/state pattern: the benchmark
harness (thread pool of VLM reads), the calibration harness (one run at
a time, camera serialized by a lock) and the training job. Config lives
server-side only (mode 0600); the API key is never returned unmasked.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from . import calibrate as cal_mod
from . import classifier, cnn_reader, dataset, golden, glyphs, paths, score
from . import config as config_mod
from . import train_cnn
from .camera import Camera, CameraError
from .config import ConfigError
from .display import CalibError, Display
from .reader import ReaderError, VlmReader, jpeg_bytes
from .textread import DEFAULT_OCR_PROMPT, ModeReader
from .vlm import VLMClient

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="Split-Flap Calib Auto")

# One physical camera: live view, camera checks and runs all serialize
# on this lock (Windows drivers fail on concurrent opens).
_camera_lock = threading.Lock()
_CHECK_LOCK_WAIT_S = 2.5
_RUN_LOCK_WAIT_S = 30.0
_COMPACT_DROP = ("modules",)
# The tool-call path's fixed production encoding (the defaults of
# reader.jpeg_bytes applied to the annotated photo).
TOOL_PATH_IMAGE = {"annotated": True, "image_max_width": 1024,
                   "image_quality": 80, "image_format": "jpeg"}


def _write_json(path: str, payload) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, path)


def _sha256_of(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return ""


def vlm_session_id(cfg: dict) -> str:
    """Stable session id derived from the API key (prompt cache)."""
    scope = str(cfg.get("llm_api_key") or cfg.get("llm_base_url") or "local")
    return uuid.uuid5(uuid.NAMESPACE_URL, "splitflap-calib-auto:" + scope).hex


def _pipeline_summary(cfg: dict) -> dict:
    """Canonical description of how a photo became characters."""
    mode = str(cfg.get("read_mode", "auto"))
    img_mode = str(cfg.get("image_mode", "strip"))
    prep = str(cfg.get("preprocess", "none"))
    parts = {
        "read_mode": mode,
        "segmentation": img_mode,
        "preprocess": prep,
        "module_count": int(cfg.get("module_count", 12)),
        "text_path": {
            "image_max_width": int(cfg.get("image_max_width", 1024)),
            "image_quality": int(cfg.get("image_quality", 80)),
            "image_format": str(cfg.get("image_format", "jpeg")),
            "ocr_prompt": cfg.get("ocr_prompt") or DEFAULT_OCR_PROMPT,
            "ocr_max_tokens": cfg.get("ocr_max_tokens") or None,
        },
        "tool_path": dict(TOOL_PATH_IMAGE),
    }
    text = parts["text_path"]
    segments = [f"read={mode}", f"seg={img_mode}", f"prep={prep}"]
    if mode in ("text", "auto"):
        segments.append(
            f"text-img={text['image_max_width']}px q{text['image_quality']} "
            f"{text['image_format']}")
        segments.append(f'prompt="{text["ocr_prompt"]}"')
        segments.append(
            f"max_tok={text['ocr_max_tokens'] or 'provider-default'}")
    if mode in ("tool", "auto"):
        segments.append("tool-img=annotated 1024px q80 jpeg")
    segments.append(f"modules={parts['module_count']}")
    return {"text": " · ".join(segments), "parts": parts}


def _page(name: str) -> HTMLResponse:
    path = os.path.join(STATIC_DIR, name)
    try:
        with open(path, encoding="utf-8") as fh:
            return HTMLResponse(fh.read())
    except OSError:
        raise HTTPException(404, f"page {name} not found")


def _golden_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except golden.GoldenError as exc:
        raise HTTPException(400, str(exc)) from exc


# -- pages ---------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def page_index():
    return _page("index.html")


@app.get("/cnn", response_class=HTMLResponse)
def page_cnn():
    return _page("cnn.html")


@app.get("/vlm", response_class=HTMLResponse)
def page_vlm():
    return _page("vlm.html")


@app.get("/golden", response_class=HTMLResponse)
def page_golden():
    return _page("golden.html")


@app.get("/train", response_class=HTMLResponse)
def page_train():
    return _page("train.html")


@app.get("/bench", response_class=HTMLResponse)
def page_bench():
    return _page("bench.html")


@app.get("/cnn-test", response_class=HTMLResponse)
def page_cnn_test():
    return _page("cnn-test.html")


@app.get("/cal-review", response_class=HTMLResponse)
def page_cal_review():
    return _page("cal-review.html")


# -- config --------------------------------------------------------------------

@app.get("/api/config")
def get_config():
    return config_mod.masked_config()


@app.post("/api/config")
def post_config(patch: dict):
    try:
        return config_mod.save_config(patch)
    except ConfigError as exc:
        raise HTTPException(400, str(exc)) from exc


# -- display / camera ----------------------------------------------------------

@app.get("/api/display")
def display_status():
    cfg = config_mod.load_config()
    host = cfg.get("display_host", "")
    if not host:
        return {"reachable": False, "error": "display host not configured"}
    try:
        display = Display(host)
        status = display.status()
        contract = display.contract()
        return {"reachable": True, "host": host, "status": status,
                "contract": contract}
    except CalibError as exc:
        return {"reachable": False, "host": host, "error": str(exc)}


def _truthy(value) -> bool:
    """Env-var style boolean: unset/empty is False, else the usual words."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _opt_float(source, key: str, cfg: dict) -> float | None:
    raw = source.get(key) if source and key in source else None
    if raw in (None, ""):
        raw = cfg.get(key)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _exposure_of(source, cfg: dict) -> float | None:
    """Resolve sensor exposure for a camera request.

    A key present in ``source`` always wins, so the UI can explicitly ask for
    driver auto-exposure (``None``/``""``/``"auto"``) instead of silently
    inheriting a manual value stored in the config. An absent key falls back
    to the stored config.
    """
    raw = source["exposure"] if source and "exposure" in source \
        else cfg.get("exposure")
    if raw is None or (isinstance(raw, str)
                       and raw.strip().lower() in ("", "auto")):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _camera_from(source, cfg: dict) -> Camera:
    return Camera(
        int(_opt_float(source, "camera_index", cfg) or cfg.get(
            "camera_index", 0)),
        width=int(cfg.get("camera_width", 1280)),
        height=int(cfg.get("camera_height", 720)),
        brightness=float(_opt_float(source, "brightness", cfg) or 50.0),
        exposure=_exposure_of(source, cfg),
        crop_percent=float(_opt_float(source, "crop_percent", cfg) or 0.0),
        warmup_s=_opt_float(source, "warmup_s", cfg))


@app.post("/api/check-camera")
def check_camera(body: dict | None = None):
    body = body or {}
    cfg = config_mod.load_config()
    if not _camera_lock.acquire(timeout=_CHECK_LOCK_WAIT_S):
        raise HTTPException(409, "camera busy (live view or a run holds it)")
    camera = None
    try:
        camera = _camera_from(body, cfg)
        camera.open()
        diag = camera.check_camera()
        diag["current_exposure"] = camera.current_exposure
        return {"ok": True, "diagnostics": diag}
    except CameraError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass
        _camera_lock.release()


@app.get("/api/camera/frame")
def camera_frame(camera_index: int | None = None,
                 brightness: float | None = None,
                 crop_percent: float | None = None,
                 exposure: str | None = None,
                 warmup_s: str | None = None):
    """One live-view frame (JPEG). Query params override the config.

    ``exposure`` accepts a number (manual sensor exposure) or the literal
    ``auto``/empty (driver auto-exposure) — the live view needs the latter.
    """
    exposure_val: float | None = None
    if exposure is not None and exposure.strip().lower() not in ("", "auto"):
        try:
            exposure_val = float(exposure)
        except ValueError:
            raise HTTPException(400, "exposure must be a number or 'auto'")
    warmup_val: float | None = None
    if warmup_s is not None and warmup_s.strip() != "":
        try:
            warmup_val = float(warmup_s)
        except ValueError:
            raise HTTPException(400, "warmup_s must be a number")
    body = {"camera_index": camera_index, "brightness": brightness,
            "crop_percent": crop_percent, "exposure": exposure_val,
            "warmup_s": warmup_val}
    cfg = config_mod.load_config()
    if not _camera_lock.acquire(timeout=_CHECK_LOCK_WAIT_S):
        raise HTTPException(409, "camera busy (a run holds it)")
    camera = None
    try:
        camera = _camera_from(body, cfg)
        camera.open(quick=True)
        frame = camera.capture()
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise HTTPException(500, "JPEG encode failed")
        return HTMLResponse(content=bytes(buf), media_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})
    except CameraError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass
        _camera_lock.release()


# -- golden set ----------------------------------------------------------------

@app.get("/api/golden")
def golden_list():
    return {"sets": golden.list_sets()}


@app.post("/api/golden/import")
def golden_import(body: dict | None = None):
    body = body or {}
    result = _golden_call(golden.create_set,
                          str(body.get("source_dir") or ""),
                          name=body.get("name") or None,
                          module_count=int(body.get("module_count") or 12),
                          image_format=str(body.get("image_format") or "auto"))
    return result


@app.get("/api/golden/{name}")
def golden_get(name: str):
    return _golden_call(golden.get_set, name)


@app.post("/api/golden/{name}/entry")
def golden_entry(name: str, body: dict | None = None):
    body = body or {}
    photo = str(body.get("photo") or "")
    if not photo:
        raise HTTPException(400, "photo is required")
    content = body.get("content")
    status = body.get("status")
    if content is None and status is None:
        raise HTTPException(400, "content or status is required")
    entry = _golden_call(golden.update_entry, name, photo,
                         content=str(content) if content is not None else None,
                         status=str(status) if status is not None else None)
    return {"entry": entry, "stats": golden.stats(golden.read_entries(name))}


@app.post("/api/golden/{name}/remove-image")
def golden_remove(name: str, body: dict | None = None):
    body = body or {}
    photo = str(body.get("photo") or "")
    if not photo:
        raise HTTPException(400, "photo is required")
    stats = _golden_call(golden.remove_image, name, photo)
    return {"ok": True, "stats": stats}


@app.delete("/api/golden/{name}")
def golden_delete(name: str):
    _golden_call(golden.delete_set, name)
    return {"ok": True}


@app.get("/api/golden/{name}/photo/{photo}")
def golden_photo(name: str, photo: str):
    if not dataset.valid_photo_name(photo):
        raise HTTPException(400, "bad photo name")
    path = golden.photo_file(name, photo)
    if path is None:
        raise HTTPException(404, "no such photo")
    media = "image/png"
    lowered = photo.lower()
    if lowered.endswith((".jpg", ".jpeg")):
        media = "image/jpeg"
    elif lowered.endswith(".webp"):
        media = "image/webp"
    elif lowered.endswith(".bmp"):
        media = "image/bmp"
    return FileResponse(path, media_type=media)


# -- models + training job ------------------------------------------------------

@app.get("/api/models")
def models_info():
    out: dict = {"caches": [], "bank": None, "cnn": None,
                 "charset": classifier.CHARSET}
    for name in glyphs.available_sets():
        summary_path = os.path.join(glyphs.cache_dir(), name + ".json")
        try:
            with open(summary_path, encoding="utf-8") as fh:
                out["caches"].append(json.load(fh))
        except (OSError, ValueError):
            out["caches"].append({"set": name})
    for key, path in (("bank", classifier.bank_path()),
                      ("cnn", train_cnn.model_path())):
        meta_path = os.path.splitext(path)[0] + ".json"
        if not os.path.isfile(path):
            continue
        entry: dict = {"file": os.path.basename(path),
                       "sha256": _sha256_of(path)[:12]}
        try:
            with open(meta_path, encoding="utf-8") as fh:
                entry["meta"] = json.load(fh)
        except (OSError, ValueError):
            entry["meta"] = {}
        out[key] = entry
    return out


class TrainJob:
    """Owns one CNN training job at a time (background thread)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"
        self.events: list[dict] = []
        self.result: dict | None = None
        self.history: list[dict] = []  # per-epoch {epoch, loss, val_acc?}
        self.run_dir = ""
        self.run_seq = 0
        self.thread: threading.Thread | None = None

    def log(self, text: str, kind: str = "log"):
        evt = {"t": time.strftime("%H:%M:%S"), "kind": kind, "text": text}
        with self.lock:
            self.events.append(evt)
            run_dir = self.run_dir
        if run_dir:
            try:
                with open(os.path.join(run_dir, "events.jsonl"), "a",
                          encoding="utf-8") as fh:
                    fh.write(json.dumps(evt, default=str) + "\n")
            except OSError:
                pass

    def state(self) -> dict:
        with self.lock:
            return {"status": self.status, "run_dir": self.run_dir,
                    "run_seq": self.run_seq, "result": self.result,
                    "history": list(self.history),
                    "event_count": len(self.events),
                    "events": self.events[-200:]}

    def events_since(self, offset: int, limit: int = 500) -> dict:
        with self.lock:
            total = len(self.events)
            offset = max(0, min(offset, total))
            return {"total": total, "offset": offset,
                    "events": self.events[offset:offset + max(1, limit)]}

    def start(self, body: dict) -> dict:
        # Parse the request BEFORE announcing a run: a bad epochs/seed
        # value must not leave the job wedged at "running" with no thread
        # behind it (every later start would 409 until a restart).
        sets = body.get("sets") or None
        try:
            epochs = max(1, min(200, int(body.get("epochs") or 30)))
            seed = int(body.get("seed") or 1)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                400, "epochs and seed must be integers") from exc
        device_req = body.get("device") or None
        with self.lock:
            if self.status == "running":
                raise HTTPException(409, "training already in progress")
            # Allocate while holding the lock but BEFORE flipping the
            # status: an OSError (full disk, unwritable CALIB_AUTO_DATA)
            # must not leave the job stuck "running" with no thread.
            try:
                run_dir = paths.alloc_run_dir("train")
            except OSError as exc:
                raise HTTPException(
                    500, f"cannot allocate a run directory: {exc}") from exc
            self.status = "running"
            self.events = []
            self.result = None
            self.history = []
            self.run_seq += 1
            self.run_dir = run_dir

        def _run():
            try:
                self.log(f"building glyph caches from the golden set "
                         f"({'all sets' if not sets else ', '.join(sets)})")
                self.log(f"device: {train_cnn.resolve_device(device_req)} "
                         f"(auto: CUDA > MPS > CPU)")
                build = glyphs.build(sets=sets)
                for name, summary in build["sets"].items():
                    self.log(f"  cache {name}: {summary['cells']} cells from "
                             f"{summary['photos']} photos"
                             + (f" ({summary['skipped']} skipped)"
                                if summary.get("skipped") else ""))
                if build["no_train_set"]:
                    self.log("  no usable entries: "
                             + ", ".join(build["no_train_set"]))
                # Train on exactly the sets that produced caches: a
                # selected-but-empty set must not fail the run, and the
                # artifact's recorded sets must match what was trained on.
                usable = [name for name, summary in build["sets"].items()
                          if summary.get("cells")]
                if not usable:
                    raise classifier.BankError(
                        "no verified golden entries to train on")
                self.log(f"training on {len(usable)} set(s): "
                         + ", ".join(usable))
                self.log(f"photo-split validation run ({epochs} epochs)")

                def on_epoch(rec: dict):
                    extra = (f" val {rec['val_acc']:.4f}"
                             if "val_acc" in rec else "")
                    self.log(f"epoch {rec['epoch']:2d}: loss "
                             f"{rec['loss']:.4f}{extra}")
                    with self.lock:
                        self.history.append(dict(rec))

                split = train_cnn.train(sets=usable, epochs=epochs,
                                        seed=seed, on_epoch=on_epoch,
                                        device=device_req)
                metrics = split["metrics"]
                if metrics:
                    self.log(f"validation: {metrics['cnn_acc']:.3f} per-cell, "
                             f"{metrics['row_exact']}/{metrics['rows']} rows "
                             f"exact, avg {metrics['row_mismatch_avg']} "
                             f"mismatches/row")
                if split["excluded"]:
                    self.log("excluded labels (not in charset): "
                             + ", ".join(f"{c!r} x{n}"
                                         for c, n in
                                         split["excluded"].items()))
                self.log(f"final run on all cells of "
                         f"{', '.join(usable)} (deployment artifact)")
                final = train_cnn.train(sets=usable, epochs=epochs,
                                        seed=seed, all_data=True,
                                        verbose=False, device=device_req)
                path = train_cnn.save_model(
                    final, train_cnn.model_path(),
                    meta={"metrics": metrics, "trained": "all-data"})
                self.log(f"saved {path}")
                device_used = (final.get("device")
                               or split.get("device") or "cpu")
                with self.lock:
                    self.result = {
                        "model": os.path.basename(path),
                        "cells": final["cells"],
                        "sets": final["sets"],
                        "metrics": metrics,
                        "excluded": split["excluded"],
                        "device": device_used,
                        "finished": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"),
                    }
                _write_json(os.path.join(run_dir, "result.json"),
                            self.result)
                with self.lock:
                    self.status = "done"
            except Exception as exc:
                self.log(f"training failed: {exc}", kind="error")
                with self.lock:
                    self.status = "failed"
                    self.result = {"error": str(exc)}

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()
        return {"status": "running", "run_dir": run_dir}


train_job = TrainJob()


@app.post("/api/train/start")
def train_start(body: dict | None = None):
    return train_job.start(body or {})


@app.get("/api/train/state")
def train_state():
    return train_job.state()


@app.get("/api/train/events")
def train_events(offset: int = 0, limit: int = 500):
    return train_job.events_since(offset, limit)


@app.get("/api/train/runs")
def train_runs(limit: int = 20):
    """Past training jobs from disk — which sets each model was built on."""
    runs_dir = paths.runs_root()
    try:
        names = sorted((n for n in os.listdir(runs_dir)
                        if n.startswith("train-")), reverse=True)
    except OSError:
        return {"runs": []}
    out = []
    for name in names[:max(1, min(limit, 100))]:
        path = os.path.join(runs_dir, name, "result.json")
        try:
            with open(path, encoding="utf-8") as fh:
                res = json.load(fh)
        except (OSError, ValueError):
            continue
        metrics = res.get("metrics") or {}
        out.append({
            "run": name,
            "finished": res.get("finished"),
            "sets": res.get("sets") or [],
            "cells": res.get("cells"),
            "model": res.get("model"),
            "val_acc": metrics.get("cnn_acc"),
            "rows": metrics.get("rows"),
            "row_exact": metrics.get("row_exact"),
            "error": res.get("error"),
        })
    return {"runs": out}


# -- benchmark harness ----------------------------------------------------------

def _make_bench_reader(cfg: dict) -> ModeReader:
    if str(cfg.get("read_mode", "auto")) == "classify":
        raise HTTPException(400, "benchmark supports VLM read modes only "
                                 "(auto/tool/text)")
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
    return ModeReader(vlm, mode=cfg.get("read_mode", "auto"),
                      ocr_prompt=cfg.get("ocr_prompt") or DEFAULT_OCR_PROMPT,
                      annotate=True,
                      image_max_width=cfg.get("image_max_width", 1024),
                      image_quality=cfg.get("image_quality", 80),
                      image_format=cfg.get("image_format", "jpeg"),
                      ocr_max_tokens=cfg.get("ocr_max_tokens") or None,
                      image_mode=cfg.get("image_mode", "strip"),
                      preprocess=cfg.get("preprocess", "none"),
                      blank_gate=bool(cfg.get("blank_gate", False)))


_bench_worker = threading.local()


def _bench_reader_for(cfg: dict) -> ModeReader:
    reader = getattr(_bench_worker, "reader", None)
    if reader is None:
        reader = _make_bench_reader(cfg)
        _bench_worker.reader = reader
    return reader


def _modules_as_dicts(reading) -> list[dict]:
    return [m.as_dict() if hasattr(m, "as_dict") else dict(m)
            for m in (reading.modules or [])]


def _compact(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in _COMPACT_DROP}


class BenchHarness:
    """Owns one benchmark run at a time (background thread) + state."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.rows: list[dict | None] = []
        self.status = "idle"
        self.report: dict | None = None
        self.dataset_desc: dict = {}
        self.pipeline: dict = {}
        self.run_dir = ""
        self.thread: threading.Thread | None = None
        self.run_seq = 0
        self.abort_event = threading.Event()
        self.t0 = 0.0

    def log(self, kind: str, text: str):
        evt = {"t": time.strftime("%H:%M:%S"), "kind": kind, "text": text}
        with self.lock:
            self.events.append(evt)
            run_dir = self.run_dir
        if run_dir:
            try:
                with open(os.path.join(run_dir, "events.jsonl"), "a",
                          encoding="utf-8") as fh:
                    fh.write(json.dumps(evt, default=str) + "\n")
            except OSError:
                pass

    def state(self) -> dict:
        with self.lock:
            done = sum(1 for r in self.rows if r is not None)
            return {"status": self.status, "total": len(self.rows),
                    "done": done, "run_dir": self.run_dir,
                    "run_seq": self.run_seq, "report": self.report,
                    "dataset": self.dataset_desc,
                    "pipeline": self.pipeline,
                    "event_count": len(self.events),
                    "events": self.events[-200:]}

    def events_since(self, offset: int, limit: int = 500) -> dict:
        with self.lock:
            total = len(self.events)
            offset = max(0, min(offset, total))
            return {"total": total, "offset": offset,
                    "events": self.events[offset:offset + max(1, limit)]}

    def rows_since(self) -> dict:
        with self.lock:
            rows = [row for row in self.rows if row is not None]
            return {"run_seq": self.run_seq, "total": len(self.rows),
                    "rows": [_compact(r) for r in rows]}

    def row_detail(self, index: int) -> dict | None:
        with self.lock:
            if 0 <= index < len(self.rows) and self.rows[index] is not None:
                return self.rows[index]
        return None

    def start(self, cfg: dict, limit: int | None = None) -> dict:
        with self.lock:
            if self.status in ("running", "aborting"):
                raise HTTPException(409, "run already in progress")
            # Allocate while holding the lock but BEFORE flipping the
            # status: an OSError (full disk, unwritable CALIB_AUTO_DATA)
            # must not leave the harness stuck "running" with no thread
            # (every later start would 409 until a restart).
            try:
                run_dir = paths.alloc_run_dir("bench")
            except OSError as exc:
                raise HTTPException(
                    500, f"cannot allocate a run directory: {exc}") from exc
            self.status = "running"
            self.events = []
            self.rows = []
            self.report = None
            self.dataset_desc = {}
            self.pipeline = _pipeline_summary(cfg)
            self.run_seq += 1
            self.abort_event = threading.Event()
            self.t0 = time.monotonic()
            self.run_dir = run_dir
        self.thread = threading.Thread(target=self._run, args=(cfg, limit),
                                       daemon=True)
        self.thread.start()
        return {"status": "running", "run_dir": run_dir}

    def abort(self) -> None:
        with self.lock:
            if self.status not in ("running", "aborting"):
                return
            self.status = "aborting"
        self.abort_event.set()

    def _run(self, cfg: dict, limit: int | None) -> None:
        started = datetime.now(timezone.utc)
        run_dir = self.run_dir
        try:
            module_count = int(cfg.get("module_count", 12))
            ds = dataset.load_dataset(cfg.get("dataset_dir", ""),
                                      cfg.get("dataset_file",
                                              dataset.GOLDEN_FILE),
                                      module_count=module_count,
                                      verified_only=cfg.get("verified_only",
                                                            True))
            if not ds.records:
                if ds.skipped_pending:
                    raise RuntimeError(
                        "no verified entries yet: "
                        f"{ds.skipped_pending} pending, 0 verified — curate "
                        "the set first")
                raise RuntimeError(
                    "dataset empty or unreadable: "
                    + ("; ".join(ds.problems) if ds.problems else ds.path))
            chosen = ds.records[:limit] if limit else ds.records
            self.log("run", f"started: {len(chosen)} rows from {ds.path}")
            with self.lock:
                self.rows = [None] * len(chosen)
                self.dataset_desc = {"dir": ds.directory, "file": ds.filename,
                                     "path": ds.path, "rows": len(chosen),
                                     "rows_in_file": len(ds.records),
                                     "kind": ds.kind,
                                     "skipped_pending": ds.skipped_pending}
            for pos, rec in enumerate(chosen):
                if not rec.runnable:  # invalid line: error row, no VLM call
                    self._emit(pos, self._error_row(pos, rec,
                                                    "; ".join(rec.fatal)))
            workers = max(1, min(config_mod.MAX_CONCURRENCY,
                                 int(cfg.get("concurrency", 1))))
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=workers) as pool:
                futures = {}
                for pos, rec in enumerate(chosen):
                    if rec.runnable and not self.abort_event.is_set():
                        futures[pool.submit(self._work, pos, rec, ds, cfg,
                                            module_count)] = pos
                for fut in concurrent.futures.as_completed(futures):
                    pos = futures[fut]
                    try:
                        row = fut.result()
                    except Exception as exc:  # worker bug: never kill the run
                        row = self._error_row(pos, chosen[pos],
                                              f"worker crashed: {exc}")
                    if row is not None:
                        self._emit(pos, row)
            status = "aborted" if self.abort_event.is_set() else "done"
            rows_now = [r for r in self._snapshot_rows() if r is not None]
            summary = score.summarize(rows_now, width=module_count)
            finished = datetime.now(timezone.utc)
            times = [r["elapsed_s"] for r in rows_now
                     if r.get("elapsed_s") is not None]
            avg_time_s = round(sum(times) / len(times), 2) if times else None
            report = {
                "app": "calib-auto",
                "run": os.path.basename(run_dir),
                "pipeline": _pipeline_summary(cfg),
                "status": status,
                "started": started.isoformat(timespec="seconds"),
                "finished": finished.isoformat(timespec="seconds"),
                "elapsed_s": round(time.monotonic() - self.t0, 1),
                "dataset": {
                    "dir": ds.directory, "file": ds.filename, "path": ds.path,
                    "kind": ds.kind,
                    "rows_in_file": len(ds.records),
                    "rows_in_run": len(chosen),
                    "verified_only": bool(cfg.get("verified_only", True)),
                    "skipped_pending": ds.skipped_pending,
                    "sha256": _sha256_of(ds.path),
                },
                "config": {
                    "llm_base_url": cfg.get("llm_base_url"),
                    "llm_model": cfg.get("llm_model"),
                    "charset": cfg.get("charset"),
                    "read_mode": cfg.get("read_mode", "auto"),
                    "ocr_prompt": cfg.get("ocr_prompt") or DEFAULT_OCR_PROMPT,
                    "image_max_width": cfg.get("image_max_width", 1024),
                    "image_quality": cfg.get("image_quality", 80),
                    "image_format": cfg.get("image_format", "jpeg"),
                    "image_mode": cfg.get("image_mode", "strip"),
                    "preprocess": cfg.get("preprocess", "none"),
                    "debug_images": bool(cfg.get("debug_images", False)),
                    "verified_only": bool(cfg.get("verified_only", True)),
                    "ocr_max_tokens": cfg.get("ocr_max_tokens", 128),
                    "module_count": module_count,
                    "concurrency": workers,
                    "limit": limit,
                },
                "counts": {
                    "total": len(chosen),
                    "done": len(rows_now),
                    "errors": summary["errors"],
                    "adjusted": summary["adjusted"],
                    "text_fallback": sum(1 for r in rows_now
                                         if r.get("fallback")),
                    "no_detect": sum(1 for r in rows_now
                                     if "no-detect" in (r.get("flags") or [])),
                    "skipped_pending": ds.skipped_pending,
                    "avg_time_s": avg_time_s,
                },
                "summary": summary,
            }
            _write_json(os.path.join(run_dir, "results.json"),
                        {"run": report["run"], "rows": rows_now})
            _write_json(os.path.join(run_dir, "report.json"), report)
            vlm, saw = summary["vlm"], summary["saw"]
            self.log("done", (
                f"{status}: model total {vlm['total']} mismatches "
                f"(avg {vlm['mean']}) vs baseline {saw['total']} "
                f"(avg {saw['mean']}) over {summary['scored_vlm']} scored "
                f"rows; {summary['errors']} error(s), "
                f"{summary['adjusted']} adjusted"))
            with self.lock:
                self.report = report
                self.status = status
        except Exception as exc:
            self.log("error", f"run failed: {exc}")
            with self.lock:
                self.status = "failed"
                self.report = {"app": "calib-auto", "status": "failed",
                               "started": started.isoformat(timespec="seconds"),
                               "reason": str(exc)}
            try:
                _write_json(os.path.join(run_dir, "report.json"),
                            self.report)
            except OSError:
                pass

    def _snapshot_rows(self) -> list[dict | None]:
        with self.lock:
            return list(self.rows)

    def _work(self, pos: int, rec, ds, cfg: dict,
              module_count: int) -> dict | None:
        if self.abort_event.is_set():
            return None
        t0 = time.monotonic()
        reader = _bench_reader_for(cfg)
        path = dataset.photo_path(ds.directory, rec.photo)
        if path is None:
            return self._error_row(pos, rec, "photo not found")
        img = cv2.imread(path)
        if img is None:
            return self._error_row(pos, rec, "photo unreadable (cv2.imread)")
        width = rec.width or module_count
        charset = cfg.get("charset") or config_mod.DEFAULT_CHARSET
        try:
            # Blind read: the golden content is NEVER shown to the model
            # (the benchmark must not leak ground truth).
            reading = reader.read(img, total=width, expected="",
                                  charset=charset, drum=charset)
        except ReaderError as exc:
            return self._error_row(pos, rec, f"reader failed: {exc}")
        scored = score.score_reading(reading.text, rec.want, rec.saw, width)
        usage = getattr(getattr(reader, "vlm", None), "last_usage", None) or {}
        mode_used = getattr(reader, "last_mode", "tool")
        fallback = bool(getattr(reader, "last_fallback", False))
        raw = getattr(reader, "last_raw", None)
        flags = list(rec.issues)
        if scored["adjusted"]:
            flags.append("adjusted")
        if reading.realigned:
            flags.append("realigned")
        if fallback:
            flags.append("text-fallback")
        if getattr(reader, "last_empty", False):
            flags.append("no-text")
        if getattr(reader, "last_no_detect", False):
            flags.append("no-detect")
        segmentation = self._segmentation_meta(reader)
        composed = self._save_composed(pos, rec, cfg, reader)
        return {
            "index": pos,
            "photo": rec.photo,
            "prefix": dataset.prefix_of(rec.photo),
            "want": rec.want,
            "saw": rec.saw,
            "read": reading.text,
            "read_mode": mode_used,
            "raw": raw[:500] if isinstance(raw, str) else None,
            "fallback": fallback,
            "want_norm": scored["want_norm"],
            "saw_norm": scored["saw_norm"],
            "read_norm": scored["read_norm"],
            "mm_vlm": scored["mm_vlm"],
            "mm_saw": scored["mm_saw"],
            "adjusted": scored["adjusted"],
            "realigned": bool(reading.realigned),
            "raw_count": int(reading.raw_count or 0),
            "warnings": list(reading.warnings or []),
            "flags": flags,
            "error": None,
            "modules": _modules_as_dicts(reading),
            "segmentation": segmentation,
            "composed": composed,
            "elapsed_s": round(time.monotonic() - t0, 2),
            "usage": usage,
        }

    @staticmethod
    def _segmentation_meta(reader) -> dict | None:
        detected = getattr(reader, "last_detected", None)
        if detected is None:
            return None
        meta: dict = {"detected": bool(detected),
                      "display": getattr(reader, "last_display", None)}
        blanks = getattr(reader, "last_blanks", None)
        if blanks is not None:
            meta["blanks"] = int(sum(1 for b in blanks if b))
        return meta

    def _save_composed(self, pos: int, rec, cfg: dict,
                       reader) -> str | None:
        if not cfg.get("debug_images"):
            return None
        composed = getattr(reader, "last_composed", None)
        if composed is None or not self.run_dir:
            return None
        directory = os.path.join(self.run_dir, "composed")
        os.makedirs(directory, exist_ok=True)
        name = f"{pos:04d}_{os.path.splitext(rec.photo)[0]}.png"
        path = os.path.join(directory, name)
        if not cv2.imwrite(path, composed):
            return None
        return os.path.relpath(path, self.run_dir)

    def _error_row(self, pos: int, rec, problem: str) -> dict:
        return {
            "index": pos, "photo": rec.photo,
            "prefix": dataset.prefix_of(rec.photo),
            "want": rec.want, "saw": rec.saw, "read": None,
            "read_mode": None, "raw": None, "fallback": False,
            "want_norm": None, "saw_norm": None, "read_norm": None,
            "mm_vlm": None, "mm_saw": None, "adjusted": False,
            "realigned": False, "raw_count": 0, "warnings": [],
            "flags": list(rec.fatal) + list(rec.issues),
            "error": problem, "modules": [],
            "segmentation": None, "composed": None,
            "elapsed_s": None, "usage": {},
        }

    def _emit(self, pos: int, row: dict) -> None:
        with self.lock:
            if pos < len(self.rows):
                self.rows[pos] = row
        if row.get("error"):
            self.log("error", f"[{pos + 1}] {row['photo']}: {row['error']}")
        else:
            self.log("read", (
                f"[{pos + 1}] {row['photo']} want {row['want']!r} "
                f"read {row['read']!r} -> {row['mm_vlm']} mm "
                f"(baseline {row['mm_saw']} mm)"))


bench = BenchHarness()


@app.post("/api/bench/start")
def bench_start(body: dict | None = None):
    body = body or {}
    cfg = config_mod.load_config()
    if not cfg.get("dataset_dir"):
        raise HTTPException(400, "pick a golden set first")
    for key in ("llm_base_url", "llm_model"):
        if not cfg.get(key):
            raise HTTPException(400, f"configure {key} first")
    _make_bench_reader(cfg)  # validates provider/key before the run starts
    limit = None
    raw_limit = body.get("limit")
    if raw_limit not in (None, ""):
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            raise HTTPException(400, "limit must be a positive integer")
        if limit <= 0:
            raise HTTPException(400, "limit must be a positive integer")
    return bench.start(cfg, limit)


@app.post("/api/bench/abort")
def bench_abort():
    bench.abort()
    return {"status": bench.state()["status"]}


@app.get("/api/bench/state")
def bench_state():
    return bench.state()


@app.get("/api/bench/events")
def bench_events(offset: int = 0, limit: int = 500):
    return bench.events_since(offset, limit)


@app.get("/api/bench/rows")
def bench_rows():
    return bench.rows_since()


@app.get("/api/bench/row/{index}")
def bench_row(index: int):
    row = bench.row_detail(index)
    if row is None:
        raise HTTPException(404, "no such row")
    return row


@app.get("/api/bench/row/{index}/composed")
def bench_row_composed(index: int):
    row = bench.row_detail(index)
    if row is None or not row.get("composed"):
        raise HTTPException(404, "no composed image for this row")
    path = os.path.join(bench.run_dir, row["composed"])
    resolved = os.path.realpath(path)
    base = os.path.realpath(bench.run_dir)
    if not resolved.startswith(base + os.sep) or not os.path.isfile(resolved):
        raise HTTPException(404, "no composed image for this row")
    return FileResponse(resolved, media_type="image/png")


@app.get("/api/bench/report")
def bench_report():
    return bench.report or {}


@app.get("/api/bench/runs")
def bench_runs(limit: int = 25):
    """Past benchmark runs from disk (headline numbers)."""
    runs_dir = paths.runs_root()
    try:
        names = sorted((n for n in os.listdir(runs_dir)
                        if n.startswith("bench-")), reverse=True)
    except OSError:
        return {"runs": []}
    out = []
    for name in names[:max(1, min(limit, 200))]:
        path = os.path.join(runs_dir, name, "report.json")
        try:
            with open(path, encoding="utf-8") as fh:
                rep = json.load(fh)
        except (OSError, ValueError):
            continue
        summary = rep.get("summary") or {}
        vlm = summary.get("vlm") or {}
        saw = summary.get("saw") or {}
        pipeline_rec = rep.get("pipeline")
        pipeline = (pipeline_rec.get("text")
                    if isinstance(pipeline_rec, dict) else pipeline_rec)
        out.append({
            "run": rep.get("run") or name,
            "status": rep.get("status"),
            "finished": rep.get("finished"),
            "elapsed_s": rep.get("elapsed_s"),
            "llm_model": (rep.get("config") or {}).get("llm_model"),
            "pipeline": pipeline,
            "dataset_dir": (rep.get("dataset") or {}).get("dir"),
            "scored_vlm": summary.get("scored_vlm"),
            "vlm_avg": vlm.get("mean"),
            "saw_avg": saw.get("mean"),
            "vlm_total": vlm.get("total"),
            "saw_total": saw.get("total"),
            "errors": (rep.get("counts") or {}).get("errors"),
            "avg_time_s": (rep.get("counts") or {}).get("avg_time_s"),
        })
    return {"runs": out}


@app.get("/api/bench/photo/{name}")
def bench_photo(name: str):
    """Serve a photo of the currently configured dataset."""
    cfg = config_mod.load_config()
    directory = str(cfg.get("dataset_dir") or "")
    path = dataset.photo_path(directory, name) if directory else None
    if path is None:
        raise HTTPException(404, "no such photo")
    return FileResponse(path, media_type="image/png")


READ_TEST_KEEP = 20


@app.post("/api/read-test")
def read_test(body: dict | None = None):
    """One-shot read of a golden photo with the configured VLM."""
    body = body or {}
    cfg = config_mod.load_config()
    set_name = str(body.get("set") or "")
    photo = str(body.get("photo") or "")
    if not set_name or not photo:
        raise HTTPException(400, "set and photo are required")
    path = golden.photo_file(set_name, photo)
    if path is None:
        raise HTTPException(404, "no such photo in that set")
    img = cv2.imread(path)
    if img is None:
        raise HTTPException(400, "photo unreadable")
    reader = _make_bench_reader(cfg)
    module_count = int(cfg.get("module_count", 12))
    charset = cfg.get("charset") or config_mod.DEFAULT_CHARSET
    t0 = time.monotonic()
    try:
        reading = reader.read(img, total=module_count, expected="",
                              charset=charset, drum=charset)
    except ReaderError as exc:
        raise HTTPException(502, f"read failed: {exc}") from exc
    elapsed = round(time.monotonic() - t0, 2)
    # Keep a copy of the tested photo for the UI (bounded history).
    test_dir = os.path.join(paths.runs_root(), "read-tests")
    os.makedirs(test_dir, exist_ok=True)
    try:
        existing = sorted(os.listdir(test_dir))
        for old in existing[:-READ_TEST_KEEP + 1]:
            try:
                os.remove(os.path.join(test_dir, old))
            except OSError:
                pass
        stamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(os.path.join(test_dir, f"{stamp}_{photo}"), img)
    except OSError:
        pass
    return {
        "set": set_name, "photo": photo, "elapsed_s": elapsed,
        "read_mode": getattr(reader, "last_mode", "tool"),
        "read": reading.text,
        "raw": getattr(reader, "last_raw", None),
        "warnings": list(reading.warnings or []),
        "modules": _modules_as_dicts(reading),
        "usage": getattr(getattr(reader, "vlm", None), "last_usage", None)
        or {},
    }


# -- CNN test harness -----------------------------------------------------------

def _make_cnn_test_reader(cfg: dict, body: dict):
    """Build the local recognizer for a test read / benchmark run."""
    raw_conf = body.get("min_conf")
    raw_margin = body.get("min_margin")
    raw_gate = body.get("blank_gate")
    blank_gate = (bool(raw_gate) if raw_gate not in (None, "")
                  else bool(cfg.get("blank_gate", False)))
    try:
        reader = cnn_reader.CnnReader(
            backend=str(body.get("backend")
                        or cfg.get("classifier_backend", "auto")),
            model_path=cfg.get("classifier_model") or None,
            min_conf=float(raw_conf if raw_conf not in (None, "")
                           else cfg.get("classifier_min_conf", 0.5)),
            min_margin=float(raw_margin if raw_margin not in (None, "")
                             else cfg.get("classifier_min_margin", 0.1)),
            blank_gate=blank_gate)
        reader.model  # fail now when the artifact is missing/corrupt
    except classifier.BankError as exc:
        raise HTTPException(400, (
            f"local classifier not usable: {exc} — train one on the "
            "Train page first")) from exc
    return reader


def _cnn_pipeline_text(info: dict, module_count: int) -> str:
    digest = (info.get("sha256") or "")[:8] or "missing"
    sets = ",".join(info.get("sets") or []) or "-"
    gate = "on" if info.get("blank_gate") else "off"
    return (f"read=classify · seg=display+canonical64 · jpeg 1024px q80 · "
            f"model={info.get('backend', '?')}:{digest} sets={sets} · "
            f"conf>={info.get('min_conf')} "
            f"margin>={info.get('min_margin')} · blank-gate={gate} · "
            f"modules={module_count}")


@app.post("/api/cnn-test/read")
def cnn_test_read(body: dict | None = None):
    """One-shot read of a golden photo with the local model."""
    body = body or {}
    cfg = config_mod.load_config()
    set_name = str(body.get("set") or "")
    photo = str(body.get("photo") or "")
    if not set_name or not photo:
        raise HTTPException(400, "set and photo are required")
    try:
        golden.get_set(set_name)
    except golden.GoldenError as exc:
        raise HTTPException(400, str(exc)) from exc
    path = golden.photo_file(set_name, photo)
    if path is None:
        raise HTTPException(404, "no such photo in that set")
    img = cv2.imread(path)
    if img is None:
        raise HTTPException(400, "photo unreadable")
    reader = _make_cnn_test_reader(cfg, body)
    module_count = int(cfg.get("module_count", 12))
    charset = cfg.get("charset") or config_mod.DEFAULT_CHARSET
    t0 = time.monotonic()
    # Same read path as calibration: the loop JPEG-encodes the photo with
    # the reader's defaults before the model sees it, so the test does too.
    try:
        reading = reader.read(jpeg_bytes(img), total=module_count,
                              expected="", charset=charset, drum=charset)
    except Exception as exc:
        raise HTTPException(502, f"read failed: {exc}") from exc
    elapsed = round(time.monotonic() - t0, 2)
    entry = None
    for item in golden.read_entries(set_name):
        if item.get("photo") == photo:
            entry = item
            break
    content = str((entry or {}).get("content") or "")
    want_norm, _ = score.normalize_text(content, module_count)
    got_norm, _ = score.normalize_text(reading.text, module_count)
    info = cnn_reader.pipeline_info(
        backend=reader.backend, model_path=reader.model_path,
        min_conf=reader.min_conf, min_margin=reader.min_margin,
        blank_gate=getattr(reader, "blank_gate", False))
    return {
        "set": set_name, "photo": photo, "elapsed_s": elapsed,
        "content": content,
        "prior_read": str((entry or {}).get("prior_read") or ""),
        "read": reading.text,
        "mismatches": (score.mismatches(got_norm, want_norm)
                       if len(content) == module_count else None),
        "warnings": list(reading.warnings or []),
        "modules": _modules_as_dicts(reading),
        "low_conf": int(getattr(reader, "last_low_conf", 0) or 0),
        "detected": getattr(reader, "last_detected", None),
        "display": getattr(reader, "last_display", None),
        "model": info,
    }


class CnnTestHarness:
    """Owns one CNN benchmark run at a time (background thread) + state."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.rows: list[dict | None] = []
        self.status = "idle"
        self.report: dict | None = None
        self.dataset_desc: dict = {}
        self.pipeline: dict = {}
        self.run_dir = ""
        self.thread: threading.Thread | None = None
        self.run_seq = 0
        self.abort_event = threading.Event()
        self.t0 = 0.0

    def log(self, kind: str, text: str):
        evt = {"t": time.strftime("%H:%M:%S"), "kind": kind, "text": text}
        with self.lock:
            self.events.append(evt)
            run_dir = self.run_dir
        if run_dir:
            try:
                with open(os.path.join(run_dir, "events.jsonl"), "a",
                          encoding="utf-8") as fh:
                    fh.write(json.dumps(evt, default=str) + "\n")
            except OSError:
                pass

    def state(self) -> dict:
        with self.lock:
            done = sum(1 for r in self.rows if r is not None)
            return {"status": self.status, "total": len(self.rows),
                    "done": done, "run_dir": self.run_dir,
                    "run_seq": self.run_seq, "report": self.report,
                    "dataset": self.dataset_desc,
                    "pipeline": self.pipeline,
                    "event_count": len(self.events),
                    "events": self.events[-200:]}

    def events_since(self, offset: int, limit: int = 500) -> dict:
        with self.lock:
            total = len(self.events)
            offset = max(0, min(offset, total))
            return {"total": total, "offset": offset,
                    "events": self.events[offset:offset + max(1, limit)]}

    def rows_since(self) -> dict:
        with self.lock:
            rows = [row for row in self.rows if row is not None]
            return {"run_seq": self.run_seq, "total": len(self.rows),
                    "rows": [_compact(r) for r in rows]}

    def row_detail(self, index: int) -> dict | None:
        with self.lock:
            if 0 <= index < len(self.rows) and self.rows[index] is not None:
                return self.rows[index]
        return None

    def start(self, cfg: dict, body: dict) -> dict:
        set_name = str(body.get("set") or "")
        if not set_name:
            raise HTTPException(400, "pick a golden set first")
        try:
            golden.get_set(set_name)
        except golden.GoldenError as exc:
            raise HTTPException(400, str(exc)) from exc
        _make_cnn_test_reader(cfg, body)  # validate before starting
        limit = None
        raw_limit = body.get("limit")
        if raw_limit not in (None, ""):
            try:
                limit = int(raw_limit)
            except (TypeError, ValueError):
                raise HTTPException(400, "limit must be a positive integer")
            if limit <= 0:
                raise HTTPException(400, "limit must be a positive integer")
        with self.lock:
            if self.status in ("running", "aborting"):
                raise HTTPException(409, "run already in progress")
            # Allocate while holding the lock but BEFORE flipping the
            # status: an OSError (full disk, unwritable CALIB_AUTO_DATA)
            # must not leave the harness stuck "running" with no thread
            # (every later start would 409 until a restart).
            try:
                run_dir = paths.alloc_run_dir("cnn")
            except OSError as exc:
                raise HTTPException(
                    500, f"cannot allocate a run directory: {exc}") from exc
            self.status = "running"
            self.events = []
            self.rows = []
            self.report = None
            self.dataset_desc = {}
            self.pipeline = {}
            self.run_seq += 1
            self.abort_event = threading.Event()
            self.t0 = time.monotonic()
            self.run_dir = run_dir
        self.thread = threading.Thread(target=self._run,
                           args=(cfg, body, set_name, limit),
                                       daemon=True)
        self.thread.start()
        return {"status": "running", "run_dir": run_dir}

    def abort(self) -> None:
        with self.lock:
            if self.status not in ("running", "aborting"):
                return
            self.status = "aborting"
        self.abort_event.set()

    def _run(self, cfg: dict, body: dict, set_name: str,
             limit: int | None) -> None:
        started = datetime.now(timezone.utc)
        run_dir = self.run_dir
        try:
            set_path = golden.set_dir(set_name)
            module_count = int(cfg.get("module_count", 12))
            ds = dataset.load_dataset(set_path, dataset.GOLDEN_FILE,
                                      module_count=module_count,
                                      verified_only=True)
            if not ds.records:
                if ds.skipped_pending:
                    raise RuntimeError(
                        "no verified entries yet: "
                        f"{ds.skipped_pending} pending, 0 verified — curate "
                        "the set first")
                raise RuntimeError(
                    "dataset empty or unreadable: "
                    + ("; ".join(ds.problems) if ds.problems else ds.path))
            chosen = ds.records[:limit] if limit else ds.records
            reader = _make_cnn_test_reader(cfg, body)
            info = cnn_reader.pipeline_info(
                backend=reader.backend, model_path=reader.model_path,
                min_conf=reader.min_conf, min_margin=reader.min_margin,
                blank_gate=getattr(reader, "blank_gate", False))
            insample = set_name in (info.get("sets") or [])
            with self.lock:
                self.pipeline = {
                    "text": _cnn_pipeline_text(info, module_count),
                    "parts": {"model": info, "module_count": module_count,
                              "image": {"max_width": 1024, "quality": 80,
                                        "format": "jpeg"}},
                }
                self.rows = [None] * len(chosen)
                self.dataset_desc = {
                    "set": set_name, "dir": ds.directory,
                    "file": ds.filename, "path": ds.path,
                    "rows": len(chosen),
                    "rows_in_file": len(ds.records),
                    "kind": ds.kind,
                    "insample": insample,
                    "skipped_pending": ds.skipped_pending,
                }
            self.log("run", f"started: {len(chosen)} rows from {ds.path}"
                     + (" (in-sample: the model was trained on "
                        "this set)" if insample else ""))
            charset = cfg.get("charset") or config_mod.DEFAULT_CHARSET
            for pos, rec in enumerate(chosen):
                if self.abort_event.is_set():
                    break
                if not rec.runnable:  # invalid line: error row, no read
                    self._emit(pos, self._error_row(pos, rec,
                                                    "; ".join(rec.fatal)))
                    continue
                self._emit(pos, self._work(pos, rec, ds, reader,
                                           module_count, charset))
            status = "aborted" if self.abort_event.is_set() else "done"
            rows_now = [r for r in self._snapshot_rows() if r is not None]
            summary = score.summarize(rows_now, width=module_count)
            finished = datetime.now(timezone.utc)
            times = [r["elapsed_s"] for r in rows_now
                     if r.get("elapsed_s") is not None]
            report = {
                "app": "calib-auto",
                "kind": "cnn-test",
                "run": os.path.basename(run_dir),
                "pipeline": self.pipeline,
                "status": status,
                "started": started.isoformat(timespec="seconds"),
                "finished": finished.isoformat(timespec="seconds"),
                "elapsed_s": round(time.monotonic() - self.t0, 1),
                "dataset": {
                    "set": set_name, "dir": ds.directory,
                    "file": ds.filename, "path": ds.path, "kind": ds.kind,
                    "rows_in_file": len(ds.records),
                    "rows_in_run": len(chosen),
                    "verified_only": True,
                    "skipped_pending": ds.skipped_pending,
                    "insample": insample,
                    "sha256": _sha256_of(ds.path),
                },
                "model": info,
                "config": {
                    "backend": reader.backend,
                    "min_conf": reader.min_conf,
                    "min_margin": reader.min_margin,
                    "module_count": module_count,
                    "charset": charset,
                    "limit": limit,
                },
                "counts": {
                    "total": len(chosen),
                    "done": len(rows_now),
                    "errors": summary["errors"],
                    "adjusted": summary["adjusted"],
                    "low_conf_rows": sum(
                        1 for r in rows_now
                        if any(str(f).startswith("low-conf")
                               for f in (r.get("flags") or []))),
                    "no_detect": sum(1 for r in rows_now
                                     if "no-detect" in (r.get("flags") or [])),
                    "avg_time_s": (round(sum(times) / len(times), 2)
                                   if times else None),
                },
                "summary": summary,
            }
            _write_json(os.path.join(run_dir, "results.json"),
                        {"run": report["run"], "rows": rows_now})
            _write_json(os.path.join(run_dir, "report.json"), report)
            model_stats = summary["vlm"]
            base_stats = summary["saw"]
            self.log("done", (
                f"{status}: model total {model_stats['total']} mismatches "
                f"(avg {model_stats['mean']}) vs baseline "
                f"{base_stats['total']} (avg {base_stats['mean']}) over "
                f"{summary['scored_vlm']} scored rows; "
                f"{summary['errors']} error(s)"
                + (" — IN-SAMPLE (set was part of training)"
                   if insample else "")))
            with self.lock:
                self.report = report
                self.status = status
        except Exception as exc:
            self.log("error", f"run failed: {exc}")
            with self.lock:
                self.status = "failed"
                self.report = {"app": "calib-auto", "kind": "cnn-test",
                               "status": "failed",
                               "started": started.isoformat(timespec="seconds"),
                               "reason": str(exc)}
            try:
                _write_json(os.path.join(run_dir, "report.json"),
                            self.report)
            except OSError:
                pass

    def _snapshot_rows(self) -> list[dict | None]:
        with self.lock:
            return list(self.rows)

    def _work(self, pos: int, rec, ds, reader, module_count: int,
              charset: str) -> dict:
        t0 = time.monotonic()
        path = dataset.photo_path(ds.directory, rec.photo)
        if path is None:
            return self._error_row(pos, rec, "photo not found")
        img = cv2.imread(path)
        if img is None:
            return self._error_row(pos, rec, "photo unreadable (cv2.imread)")
        width = rec.width or module_count
        try:
            # Same read path as calibration: JPEG-encode like the loop
            # does (reader defaults), then classify the decoded photo.
            reading = reader.read(jpeg_bytes(img), total=width, expected="",
                                  charset=charset, drum=charset)
        except Exception as exc:
            return self._error_row(pos, rec, f"read failed: {exc}")
        scored = score.score_reading(reading.text, rec.want, rec.saw, width)
        flags = list(rec.issues)
        if scored["adjusted"]:
            flags.append("adjusted")
        low = int(getattr(reader, "last_low_conf", 0) or 0)
        if low:
            flags.append(f"low-conf:{low}")
        if getattr(reader, "last_no_detect", False):
            flags.append("no-detect")
        return {
            "index": pos,
            "photo": rec.photo,
            "prefix": dataset.prefix_of(rec.photo),
            "want": rec.want,
            "saw": rec.saw,
            "read": reading.text,
            "want_norm": scored["want_norm"],
            "saw_norm": scored["saw_norm"],
            "read_norm": scored["read_norm"],
            "mm_vlm": scored["mm_vlm"],
            "mm_saw": scored["mm_saw"],
            "adjusted": scored["adjusted"],
            "raw_count": int(reading.raw_count or 0),
            "warnings": list(reading.warnings or []),
            "flags": flags,
            "error": None,
            "modules": _modules_as_dicts(reading),
            "low_conf": low,
            "detected": getattr(reader, "last_detected", None),
            "elapsed_s": round(time.monotonic() - t0, 3),
        }

    def _error_row(self, pos: int, rec, problem: str) -> dict:
        return {
            "index": pos, "photo": rec.photo,
            "prefix": dataset.prefix_of(rec.photo),
            "want": rec.want, "saw": rec.saw, "read": None,
            "want_norm": None, "saw_norm": None, "read_norm": None,
            "mm_vlm": None, "mm_saw": None, "adjusted": False,
            "raw_count": 0, "warnings": [],
            "flags": list(rec.fatal) + list(rec.issues),
            "error": problem, "modules": [], "low_conf": 0,
            "detected": None, "elapsed_s": None,
        }

    def _emit(self, pos: int, row: dict) -> None:
        with self.lock:
            if pos < len(self.rows):
                self.rows[pos] = row
        if row.get("error"):
            self.log("error", f"[{pos + 1}] {row['photo']}: {row['error']}")
        else:
            self.log("read", (
                f"[{pos + 1}] {row['photo']} golden {row['want']!r} "
                f"read {row['read']!r} -> {row['mm_vlm']} mm "
                f"(baseline {row['mm_saw']} mm)"))


cnn_test = CnnTestHarness()


@app.post("/api/cnn-test/start")
def cnn_test_start(body: dict | None = None):
    cfg = config_mod.load_config()
    return cnn_test.start(cfg, body or {})


@app.post("/api/cnn-test/abort")
def cnn_test_abort():
    cnn_test.abort()
    return {"status": cnn_test.state()["status"]}


@app.get("/api/cnn-test/state")
def cnn_test_state():
    return cnn_test.state()


@app.get("/api/cnn-test/events")
def cnn_test_events(offset: int = 0, limit: int = 500):
    return cnn_test.events_since(offset, limit)


@app.get("/api/cnn-test/rows")
def cnn_test_rows():
    return cnn_test.rows_since()


@app.get("/api/cnn-test/row/{index}")
def cnn_test_row(index: int):
    row = cnn_test.row_detail(index)
    if row is None:
        raise HTTPException(404, "no such row")
    return row


@app.get("/api/cnn-test/report")
def cnn_test_report():
    return cnn_test.report or {}


@app.get("/api/cnn-test/runs")
def cnn_test_runs(limit: int = 25):
    """Past CNN test runs from disk (headline numbers)."""
    runs_dir = paths.runs_root()
    try:
        names = sorted((n for n in os.listdir(runs_dir)
                        if n.startswith("cnn-")), reverse=True)
    except OSError:
        return {"runs": []}
    out = []
    for name in names[:max(1, min(limit, 200))]:
        path = os.path.join(runs_dir, name, "report.json")
        try:
            with open(path, encoding="utf-8") as fh:
                rep = json.load(fh)
        except (OSError, ValueError):
            continue
        summary = rep.get("summary") or {}
        model_stats = summary.get("vlm") or {}
        base_stats = summary.get("saw") or {}
        dataset_rec = rep.get("dataset") or {}
        model_rec = rep.get("model") or {}
        pipeline_rec = rep.get("pipeline")
        pipeline = (pipeline_rec.get("text")
                    if isinstance(pipeline_rec, dict) else pipeline_rec)
        out.append({
            "run": rep.get("run") or name,
            "status": rep.get("status"),
            "finished": rep.get("finished"),
            "elapsed_s": rep.get("elapsed_s"),
            "set": dataset_rec.get("set") or dataset_rec.get("dir"),
            "insample": dataset_rec.get("insample"),
            "backend": model_rec.get("backend"),
            "sha256": model_rec.get("sha256"),
            "scored_vlm": summary.get("scored_vlm"),
            "vlm_avg": model_stats.get("mean"),
            "saw_avg": base_stats.get("mean"),
            "vlm_total": model_stats.get("total"),
            "saw_total": base_stats.get("total"),
            "errors": (rep.get("counts") or {}).get("errors"),
            "avg_time_s": (rep.get("counts") or {}).get("avg_time_s"),
            "pipeline": pipeline,
        })
    return {"runs": out}


# -- calibration harness --------------------------------------------------------

def _make_cal_reader(approach: str, cfg: dict):
    """Build the recognizer for a calibration run (CNN or VLM)."""
    if approach == "cnn":
        try:
            reader = cnn_reader.CnnReader(
                backend=cfg.get("classifier_backend", "auto"),
                model_path=cfg.get("classifier_model") or None,
                min_conf=float(cfg.get("classifier_min_conf", 0.5)),
                min_margin=float(cfg.get("classifier_min_margin", 0.1)),
                blank_gate=bool(cfg.get("blank_gate", False)))
            reader.model  # fail now when the artifact is missing/corrupt
        except classifier.BankError as exc:
            raise HTTPException(400, (
                f"local classifier not usable: {exc} — train one on the "
                "Train page first")) from exc
        return reader
    if approach == "vlm":
        for key in ("llm_base_url", "llm_model"):
            if not cfg.get(key):
                raise HTTPException(400, f"configure {key} first")
        host = (urlparse(str(cfg["llm_base_url"])).hostname or "").lower()
        if not cfg.get("llm_api_key") and host not in (
                "localhost", "127.0.0.1", "0.0.0.0", "::1"):
            raise HTTPException(400, "configure llm_api_key first")
        vlm = VLMClient(cfg["llm_base_url"], cfg["llm_model"],
                        cfg.get("llm_api_key", ""),
                        session_id=vlm_session_id(cfg))
        return VlmReader(vlm, annotate=True)
    raise HTTPException(400, "approach must be cnn or vlm")


class CalHarness:
    """Owns one calibration run at a time (background thread) + state."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.photos: list[str] = []
        self.status = "idle"
        self.mode = "full"
        self.approach = "cnn"
        self.phases: list[str] = []
        self.report: dict | None = None
        self.run_dir = ""
        self.calibrator: cal_mod.Calibrator | None = None
        self.thread: threading.Thread | None = None
        self.run_seq = 0
        # An abort can arrive after the run is "running" but before the
        # run thread has constructed the calibrator (display probe, camera
        # open, camera check): remember it and apply it on construction.
        self.pending_abort = False

    def log(self, event: dict):
        with self.lock:
            self.events.append(event)
            photo = event.get("photo")
            if photo and photo not in self.photos:
                self.photos.append(photo)
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
                    "approach": self.approach,
                    "phases": list(self.phases),
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

    def start(self, cfg: dict, body: dict) -> dict:
        approach = str(body.get("approach") or "cnn").strip().lower()
        if approach not in ("cnn", "vlm"):
            raise HTTPException(400, "approach must be cnn or vlm")
        try:
            phases = cal_mod.Calibrator._normalize_phases(body.get("phases"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        mode = str(body.get("mode") or cfg.get("mode") or "full")
        if mode not in ("dry-run", "full"):
            raise HTTPException(400, "mode must be dry-run or full")
        # Validate the recognizer before touching the display.
        _make_cal_reader(approach, cfg)
        with self.lock:
            if self.status in ("running", "aborting"):
                raise HTTPException(409, "run already in progress")
            # Allocate while holding the lock but BEFORE flipping the
            # status: an OSError (full disk, unwritable CALIB_AUTO_DATA)
            # must not leave the harness stuck "running" with no thread
            # (every later start would 409 until a restart, and abort()
            # would only move it to "aborting", which is also guarded).
            try:
                run_dir = paths.alloc_run_dir("cal")
            except OSError as exc:
                raise HTTPException(
                    500, f"cannot allocate a run directory: {exc}") from exc
            self.status = "running"
            self.pending_abort = False
            self.mode = mode
            self.approach = approach
            self.phases = phases
            self.events = []
            self.photos = []
            self.run_seq += 1
            self.report = None
            self.run_dir = run_dir

        def _run():
            camera = None
            display = None
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
                    camera = _camera_from(None, cfg)
                    camera.open()
                    try:
                        camera.check_camera()
                    except CameraError as exc:
                        self.log({"t": "", "kind": "error",
                                  "text": f"camera check failed: {exc}",
                                  "photo": None})
                        with self.lock:
                            self.status = "failed"
                            self.report = {
                                "result": "needs-human",
                                "reason": f"camera check failed: {exc}"}
                        return
                    reader = _make_cal_reader(approach, cfg)
                    context = {
                        "approach": approach,
                        "display_host": cfg.get("display_host"),
                        "camera_index": cfg.get("camera_index"),
                        "brightness": cfg.get("brightness"),
                        "exposure": cfg.get("exposure"),
                        "crop_percent": cfg.get("crop_percent"),
                        "warmup_s": cfg.get("warmup_s"),
                    }
                    if approach == "vlm":
                        context.update({
                            "llm_base_url": cfg.get("llm_base_url"),
                            "llm_model": cfg.get("llm_model"),
                            "annotate": True,
                        })
                    else:
                        context.update(cnn_reader.pipeline_info(
                            backend=cfg.get("classifier_backend", "auto"),
                            model_path=cfg.get("classifier_model") or None,
                            min_conf=float(
                                cfg.get("classifier_min_conf", 0.5)),
                            min_margin=float(
                                cfg.get("classifier_min_margin", 0.1)),
                            blank_gate=bool(cfg.get("blank_gate", False))))
                    calib = cal_mod.Calibrator(
                        display, camera, reader, photo_dir=run_dir,
                        dwell_ms=int(cfg.get("dwell_ms", 800)),
                        timeout_s=float(cfg.get("timeout_s", 60.0)),
                        min_confidence=float(cfg.get("min_confidence", 0.6)),
                        exhaustive=bool(cfg.get("exhaustive", False)),
                        mode=mode, on_event=self.log,
                        max_seconds=float(cfg.get("max_seconds", 3600.0)),
                        phases=phases,
                        run_context=context)
                    with self.lock:
                        self.calibrator = calib
                        if self.pending_abort:
                            # Abort posted while the calibrator did not
                            # exist yet (display probe / camera open):
                            # forward it now instead of losing it.
                            calib.abort()
                    self.report = calib.run()
                    with self.lock:
                        self.report = calib.report
                        self.status = "done"
                finally:
                    _camera_lock.release()
            except (CameraError, HTTPException) as exc:
                try:
                    Display(cfg["display_host"]).hold(False)
                except Exception:
                    pass
                detail = exc.detail if isinstance(exc, HTTPException) \
                    else str(exc)
                self.log({"t": "", "kind": "error",
                          "text": f"camera error: {detail}", "photo": None})
                with self.lock:
                    self.status = "failed"
                    self.report = {"result": "needs-human",
                                   "reason": f"camera error: {detail}"}
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
        """Request an abort of the live run (no-op when idle)."""
        with self.lock:
            if self.status not in ("running", "aborting"):
                return
            self.pending_abort = True
            if self.calibrator:
                self.calibrator.abort()
            if self.status == "running":
                self.status = "aborting"


cal = CalHarness()


@app.post("/api/cal/start")
def cal_start(body: dict | None = None):
    cfg = config_mod.load_config()
    return cal.start(cfg, body or {})


@app.post("/api/cal/abort")
def cal_abort():
    cal.abort()
    return {"status": cal.state()["status"]}


@app.get("/api/cal/state")
def cal_state():
    return cal.state()


@app.get("/api/cal/events")
def cal_events(offset: int = 0, limit: int = 500):
    return cal.events_since(offset, limit)


@app.get("/api/cal/report")
def cal_report():
    return cal.report or {}


@app.get("/api/cal/photo/{name}")
def cal_photo(name: str):
    if not dataset.valid_photo_name(name):
        raise HTTPException(400, "bad photo name")
    if not cal.run_dir:
        raise HTTPException(404, "no run yet")
    path = os.path.realpath(os.path.join(cal.run_dir, name))
    base = os.path.realpath(cal.run_dir)
    if not path.startswith(base + os.sep) or not os.path.isfile(path):
        raise HTTPException(404, "no such photo")
    return FileResponse(path, media_type="image/png")


def _cal_run_dir(name: str) -> str:
    if (not name.startswith("cal-") or not dataset.valid_photo_name(name + ".png")
            or "/" in name or "\\" in name):
        raise HTTPException(400, "bad calibration run name")
    path = os.path.realpath(os.path.join(paths.runs_root(), name))
    base = os.path.realpath(paths.runs_root())
    if not path.startswith(base + os.sep) or not os.path.isdir(path):
        raise HTTPException(404, "no such calibration run")
    return path


@app.get("/api/cal/runs")
def cal_runs(limit: int = 100):
    """List completed calibration runs that have a persisted report."""
    root = paths.runs_root()
    try:
        names = sorted((n for n in os.listdir(root) if n.startswith("cal-")),
                       reverse=True)
    except OSError:
        return {"runs": []}
    out = []
    for name in names[:max(1, min(limit, 200))]:
        report_path = os.path.join(root, name, "report.json")
        try:
            with open(report_path, encoding="utf-8") as fh:
                report = json.load(fh)
        except (OSError, ValueError):
            continue
        frames = report.get("frames") or []
        out.append({"run": name, "result": report.get("result"),
                    "reason": report.get("reason"),
                    "approach": (report.get("config") or {}).get("approach"),
                    "frames": len(frames),
                    "started": report.get("started"),
                    "finished": report.get("finished"),
                    "has_boxes": any(f.get("boxes") for f in frames
                                      if isinstance(f, dict))})
    return {"runs": out}


def _recreate_cal_geometry(run_dir: str, frame: dict, report: dict) -> dict:
    """Recreate CNN geometry for reports written before boxes were stored."""
    if frame.get("boxes") or (report.get("config") or {}).get("approach") != "cnn":
        return frame
    photo = str(frame.get("photo") or "")
    path = os.path.join(run_dir, photo)
    module_count = int((report.get("fleet") or {}).get("totalModules")
                       or (report.get("config") or {}).get("module_count")
                       or 12)
    image = cv2.imread(path) if dataset.valid_photo_name(photo) else None
    if image is None:
        return frame
    try:
        encoded = jpeg_bytes(image)
        decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8),
                               cv2.IMREAD_COLOR)
        display = cnn_reader.segment.find_display(decoded)
        if display is None:
            return frame
        item = dict(frame)
        item["boxes"] = [list(box) for box in cnn_reader.segment.module_boxes(
            decoded, display, module_count)]
        item["display"] = display.as_dict()
        item["geometry_image_size"] = [int(decoded.shape[1]),
                                        int(decoded.shape[0])]
        item["boxes_recreated"] = True
        return item
    except (cv2.error, ValueError, TypeError):
        return frame


@app.get("/api/cal/run/{run}/frames")
def cal_run_frames(run: str):
    run_dir = _cal_run_dir(run)
    try:
        with open(os.path.join(run_dir, "report.json"), encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, ValueError):
        raise HTTPException(404, "no report for calibration run")
    frames = []
    for index, frame in enumerate(report.get("frames") or []):
        if not isinstance(frame, dict) or not frame.get("photo"):
            continue
        item = _recreate_cal_geometry(run_dir, frame, report)
        item = dict(item)
        item["index"] = index
        frames.append(item)
    return {"run": run, "approach": (report.get("config") or {}).get("approach"),
            "result": report.get("result"), "reason": report.get("reason"),
            "frames": frames}


@app.get("/api/cal/run/{run}/photo/{name}")
def cal_run_photo(run: str, name: str):
    run_dir = _cal_run_dir(run)
    if not dataset.valid_photo_name(name):
        raise HTTPException(400, "bad photo name")
    path = os.path.realpath(os.path.join(run_dir, name))
    if not path.startswith(run_dir + os.sep) or not os.path.isfile(path):
        raise HTTPException(404, "no such photo")
    return FileResponse(path, media_type="image/png")


@app.post("/api/cal/restore-snapshot")
def cal_restore():
    """Roll the display's offsets back to the run-start snapshot."""
    with cal.lock:
        # Refuse while a run holds the display: a mid-run rollback reverts
        # committed offsets while the calibrator still tracks its own
        # overlay/residue belief, so the next preview commit would persist
        # from a stale base and write a wrong absolute offset to NVS.
        if cal.status in ("running", "aborting"):
            raise HTTPException(
                409, "run in progress; restore once it has finished")
        run_dir = cal.run_dir
    if not run_dir:
        raise HTTPException(404, "no run yet")
    snapshot_path = os.path.join(run_dir, "snapshot.json")
    try:
        with open(snapshot_path, encoding="utf-8") as fh:
            snapshot = json.load(fh)
    except (OSError, ValueError):
        raise HTTPException(404, "no snapshot for the last run")
    cfg = config_mod.load_config()
    try:
        display = Display(cfg["display_host"])
        display.restore(snapshot)
        try:
            display.reload()
        except CalibError:
            pass
        return {"ok": True}
    except CalibError as exc:
        raise HTTPException(502, str(exc)) from exc


def main():
    import uvicorn

    host = os.environ.get("CALIB_AUTO_HOST", "127.0.0.1")
    port = int(os.environ.get("CALIB_AUTO_PORT", "8004"))
    # The UI polls the live view continuously, so the default access log
    # is pure noise. Keep it off unless explicitly asked for.
    access_log = _truthy(os.environ.get("CALIB_AUTO_ACCESS_LOG"))
    log_level = os.environ.get("CALIB_AUTO_LOG_LEVEL", "info").lower()
    if log_level not in ("critical", "error", "warning", "info", "debug",
                         "trace"):
        log_level = "warning"
    uvicorn.run(app, host=host, port=port, access_log=access_log,
                log_level=log_level)


if __name__ == "__main__":
    main()
