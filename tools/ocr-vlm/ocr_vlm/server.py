"""OCR benchmark web app: VLM reads vs the recorded baseline.

Serve a small UI that points at a directory holding a ``reads.jsonl``
dataset (one ``{photo, want, saw}`` record per line, photos beside it),
send every photo to the configured VLM, and score the answer position by
position against ``want``. The ``saw`` field is scored identically and
shown as the baseline, so one run answers "is this provider/model better
than the reader that produced saw?".

Photos are read either through calib-vlm's per-module ``report_reading``
tool call (read_mode "tool"), through a plain OCR prompt (read_mode
"text", for models without function calling such as PaddleOCR-VL), or
auto-probed between the two (read_mode "auto", the default) — see
``ocr_vlm/textread.py``.

The read is BLIND: the reader is called with ``expected=""`` so the
ground truth never influences reconciliation or alignment (calib-vlm's
calibration loop passes the commanded frame there; a benchmark must
not). The API key lives server-side only (env or data dir, mode 0600).
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
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse

from calib_vlm.reader import ReaderError
from calib_vlm.vlm import VLMClient

from . import classify_reader
from . import curate
from . import segment
from .dataset import (Record, describe, discover, load_dataset, photo_path,
                      prefix_of, valid_photo_name)
from .score import score_reading, summarize
from .textread import DEFAULT_OCR_PROMPT, ModeReader

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# Charset 48 of the drum contract (tools/calib/calib/contract.json).
DEFAULT_CHARSET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':?!.-/$@#%"
DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_MODEL = "deepseek-v4-flash-vision-exp"
MAX_CONCURRENCY = 8

# Compact row keys served to the table; the full row (with per-module
# detail) is one fetch away via /api/run/row/{index}.
_COMPACT_DROP = ("modules",)


def data_dir() -> str:
    return os.environ.get("OCR_VLM_DATA", os.path.join(os.getcwd(), "data"))


def config_path() -> str:
    return os.path.join(data_dir(), "config.json")


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


def vlm_session_id(cfg: dict) -> str:
    """Stable OpenCode session id derived from the API key (prompt cache)."""
    scope = str(cfg.get("llm_api_key") or cfg.get("llm_base_url") or "local")
    return uuid.uuid5(uuid.NAMESPACE_URL, "splitflap-ocr-vlm:" + scope).hex


# The tool-call path's fixed production encoding (the defaults of
# calib_vlm.reader.jpeg_bytes applied to the annotated photo). Recorded so
# a report says exactly what was sent, even though the text path's image
# settings do not apply to it.
TOOL_PATH_IMAGE = {"annotated": True, "image_max_width": 1024,
                   "image_quality": 80, "image_format": "jpeg"}


def pipeline_summary(cfg: dict) -> dict:
    """Canonical description of how a photo became characters.

    ``parts`` is the machine-readable record stored in every report so a
    run can be recreated in code; ``text`` is the one-line form the UI
    shows. The two reader paths encode images differently on purpose —
    the tool path keeps production parity (annotated module grid, 1024 px
    q80 JPEG), the text path honours the configured image settings.
    """
    mode = str(cfg.get("read_mode", "auto"))
    seg = str(cfg.get("image_mode", "strip"))
    prep = str(cfg.get("preprocess", "none"))
    classifier_part = None
    if mode == "classify":
        classifier_part = classify_reader.pipeline_info(cfg, data_dir())
        seg = "display+canonical64"
        prep = "none"
    parts = {
        "read_mode": mode,
        "segmentation": seg,
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
    if classifier_part is not None:
        parts["classifier"] = classifier_part
    text = parts["text_path"]
    segments = [f"read={mode}", f"seg={seg}", f"prep={prep}"]
    if mode in ("text", "auto"):
        segments.append(
            f"text-img={text['image_max_width']}px q{text['image_quality']} "
            f"{text['image_format']}")
        segments.append(f'prompt="{text["ocr_prompt"]}"')
        segments.append(
            f"max_tok={text['ocr_max_tokens'] or 'provider-default'}")
    if mode in ("tool", "auto"):
        segments.append("tool-img=annotated 1024px q80 jpeg")
    if classifier_part is not None:
        digest = (classifier_part.get("sha256") or "")[:8] or "missing"
        sets = ",".join(classifier_part.get("sets") or []) or "-"
        segments.append(
            f"model={classifier_part.get('backend', '?')}:{digest} "
            f"sets={sets}")
        segments.append(f"conf>={classifier_part.get('min_conf')} "
                        f"margin>={classifier_part.get('min_margin')}")
    segments.append(f"modules={parts['module_count']}")
    return {"text": " · ".join(segments), "parts": parts}


def _int_or(value, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def _bool_or(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def load_config() -> dict:
    cfg: dict = {}
    try:
        with open(config_path(), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        pass
    env_map = {"dataset_dir": "OCR_VLM_DATASET_DIR",
               "dataset_file": "OCR_VLM_DATASET_FILE",
               "llm_base_url": "LLM_BASE_URL", "llm_model": "LLM_MODEL",
               "llm_api_key": "LLM_API_KEY"}
    for key, env in env_map.items():
        if env in os.environ and os.environ[env]:
            cfg[key] = os.environ[env]
    cfg.setdefault("dataset_dir", "")
    cfg.setdefault("dataset_file", "reads.jsonl")
    cfg.setdefault("charset", DEFAULT_CHARSET)
    cfg.setdefault("read_mode", "auto")
    cfg.setdefault("ocr_prompt", DEFAULT_OCR_PROMPT)
    cfg.setdefault("classifier_backend", "auto")
    cfg.setdefault("classifier_model", "")
    for key, default in (("classifier_min_conf", 0.5),
                         ("classifier_min_margin", 0.1)):
        try:
            cfg[key] = max(0.0, min(1.0, float(cfg.get(key, default))))
        except (TypeError, ValueError):
            cfg[key] = default
    mode = str(cfg.get("image_mode") or "strip").strip().lower()
    cfg["image_mode"] = mode if mode in (
        ("strip", "cells") + segment.DETECTED_MODES) else "strip"
    style = str(cfg.get("preprocess") or "none").strip().lower()
    cfg["preprocess"] = style if style in segment.PREPROCESS_STYLES else "none"
    cfg["debug_images"] = _bool_or(cfg.get("debug_images"), False)
    cfg["verified_only"] = _bool_or(cfg.get("verified_only"), True)
    cfg.setdefault("llm_base_url", DEFAULT_BASE_URL)
    cfg.setdefault("llm_model", DEFAULT_MODEL)
    cfg.setdefault("image_format", "jpeg")
    cfg["module_count"] = _int_or(cfg.get("module_count"), 12, 1, 64)
    cfg["concurrency"] = _int_or(cfg.get("concurrency"), 1, 1, MAX_CONCURRENCY)
    cfg["image_max_width"] = _int_or(cfg.get("image_max_width"), 1024, 256, 4096)
    cfg["image_quality"] = _int_or(cfg.get("image_quality"), 80, 1, 100)
    cfg["ocr_max_tokens"] = _int_or(cfg.get("ocr_max_tokens"), 128, 0, 4096)
    return cfg


def save_config(patch: dict) -> dict:
    os.makedirs(data_dir(), exist_ok=True)
    stored: dict = {}
    try:
        with open(config_path(), encoding="utf-8") as fh:
            stored = json.load(fh)
    except (OSError, ValueError):
        pass
    for key in ("dataset_dir", "dataset_file", "llm_base_url", "llm_model"):
        # dataset_dir is deliberately settable to "" (unconfigured state).
        if key in patch and patch[key] is not None:
            stored[key] = str(patch[key]).strip()
    if not stored.get("dataset_file"):
        stored["dataset_file"] = "reads.jsonl"
    if "charset" in patch and patch["charset"] is not None:
        stored["charset"] = str(patch["charset"]) or DEFAULT_CHARSET
    if "read_mode" in patch and patch["read_mode"] not in (None, ""):
        mode = str(patch["read_mode"]).strip().lower()
        if mode not in ("auto", "tool", "text", "classify"):
            raise HTTPException(
                400, "read_mode must be auto, tool, text or classify")
        stored["read_mode"] = mode
    if "classifier_backend" in patch and patch["classifier_backend"] \
            not in (None, ""):
        backend = str(patch["classifier_backend"]).strip().lower()
        if backend not in classify_reader.BACKENDS:
            raise HTTPException(
                400, "classifier_backend must be auto, bank or cnn")
        stored["classifier_backend"] = backend
    if "classifier_model" in patch and patch["classifier_model"] is not None:
        stored["classifier_model"] = str(patch["classifier_model"]).strip()
    for key in ("classifier_min_conf", "classifier_min_margin"):
        if key in patch and patch[key] not in (None, ""):
            try:
                value = float(patch[key])
            except (TypeError, ValueError):
                raise HTTPException(400, f"{key} must be a number")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise HTTPException(400, f"{key} must be between 0 and 1")
            stored[key] = value
    if "ocr_prompt" in patch and patch["ocr_prompt"] is not None:
        stored["ocr_prompt"] = (str(patch["ocr_prompt"]).strip()
                                or DEFAULT_OCR_PROMPT)
    if "image_format" in patch and patch["image_format"] not in (None, ""):
        fmt = str(patch["image_format"]).strip().lower()
        if fmt not in ("jpeg", "png"):
            raise HTTPException(400, "image_format must be jpeg or png")
        stored["image_format"] = fmt
    if "image_mode" in patch and patch["image_mode"] not in (None, ""):
        mode = str(patch["image_mode"]).strip().lower()
        if mode not in ("strip", "cells") + segment.DETECTED_MODES:
            raise HTTPException(
                400, "image_mode must be strip, cells, montage, "
                     "cells-detect or strip-detect")
        stored["image_mode"] = mode
    if "preprocess" in patch and patch["preprocess"] not in (None, ""):
        style = str(patch["preprocess"]).strip().lower()
        if style not in segment.PREPROCESS_STYLES:
            raise HTTPException(
                400, "preprocess must be none, contrast, invert or binary")
        stored["preprocess"] = style
    if "debug_images" in patch and patch["debug_images"] is not None:
        stored["debug_images"] = _bool_or(patch["debug_images"], False)
    if "verified_only" in patch and patch["verified_only"] is not None:
        stored["verified_only"] = _bool_or(patch["verified_only"], True)
    for key, lo, hi in (("module_count", 1, 64),
                        ("concurrency", 1, MAX_CONCURRENCY),
                        ("image_max_width", 256, 4096),
                        ("image_quality", 1, 100),
                        ("ocr_max_tokens", 0, 4096)):
        if key in patch and patch[key] not in (None, ""):
            try:
                value = float(patch[key])
            except (TypeError, ValueError):
                raise HTTPException(400, f"{key} must be a number")
            if not math.isfinite(value):
                raise HTTPException(400, f"{key} must be a number")
            stored[key] = int(max(float(lo), min(float(hi), value)))
    if patch.get("llm_api_key"):
        stored["llm_api_key"] = patch["llm_api_key"]
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


# -- VLM client construction --------------------------------------------------

_worker_state = threading.local()


def _make_classify_reader(cfg: dict):
    """Reader for read_mode=classify: local model, no provider."""
    from . import classifier

    try:
        reader = classify_reader.ClassifyReader(
            backend=cfg.get("classifier_backend", "auto"),
            model_path=cfg.get("classifier_model") or None,
            data_dir=data_dir(),
            min_conf=cfg.get("classifier_min_conf", 0.5),
            min_margin=cfg.get("classifier_min_margin", 0.1))
        reader.model  # fail now, not mid-run, when the artifact is bad
    except classifier.BankError as exc:
        raise HTTPException(400, (
            f"local classifier not usable: {exc} — build one first: "
            "`uv run python -m ocr_vlm.train_cnn` (CNN) or "
            "`uv run python -m ocr_vlm.classifier build` (bank)")) from exc
    return reader


def _make_reader(cfg: dict) -> ModeReader:
    """Build one reader (one HTTP session) — one per worker thread.

    A seam for tests: monkeypatch this to hand the harness a fake reader.
    """
    if str(cfg.get("read_mode", "auto")) == "classify":
        return _make_classify_reader(cfg)
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
    # annotate=True: the model sees the module grid exactly like the
    # production calib-vlm reads that produced the `saw` baseline.
    return ModeReader(vlm, mode=cfg.get("read_mode", "auto"),
                      ocr_prompt=cfg.get("ocr_prompt") or DEFAULT_OCR_PROMPT,
                      annotate=True,
                      image_max_width=cfg.get("image_max_width", 1024),
                      image_quality=cfg.get("image_quality", 80),
                      image_format=cfg.get("image_format", "jpeg"),
                      ocr_max_tokens=cfg.get("ocr_max_tokens") or None,
                      image_mode=cfg.get("image_mode", "strip"),
                      preprocess=cfg.get("preprocess", "none"))


def _reader_for(cfg: dict) -> VlmReader:
    reader = getattr(_worker_state, "reader", None)
    if reader is None:
        reader = _make_reader(cfg)
        _worker_state.reader = reader
    return reader


def _modules_as_dicts(reading) -> list[dict]:
    return [m.as_dict() if hasattr(m, "as_dict") else dict(m)
            for m in (reading.modules or [])]


def _compact(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in _COMPACT_DROP}


# -- Run harness ---------------------------------------------------------------

class Harness:
    """Owns one benchmark run at a time (background thread) + shared state."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events: list[dict] = []
        self.rows: list[dict | None] = []  # full rows, index-aligned
        self.status = "idle"
        self.report: dict | None = None
        self.run_dir = ""
        self.run_seq = 0
        self.dataset_desc: dict = {}
        self.pipeline: dict = {}
        self.abort_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.t0 = 0.0

    def log(self, kind: str, text: str) -> None:
        event = {"t": datetime.now().strftime("%H:%M:%S"), "kind": kind,
                 "text": text,
                 "elapsed": round(time.monotonic() - self.t0, 1)
                 if self.t0 else None}
        with self.lock:
            self.events.append(event)
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
            done = sum(1 for r in self.rows if r is not None)
            errors = sum(1 for r in self.rows if r and r.get("error"))
            summary = (self.report or {}).get("summary")
            return {"status": self.status, "run_seq": self.run_seq,
                    "total": len(self.rows), "done": done, "errors": errors,
                    "event_count": len(self.events), "run_dir": self.run_dir,
                    "summary": summary, "dataset": self.dataset_desc,
                    "pipeline": self.pipeline}

    def events_since(self, offset: int, limit: int = 500) -> dict:
        with self.lock:
            total = len(self.events)
            offset = max(0, min(offset, total))
            limit = max(1, min(limit, 2000))
            return {"total": total, "offset": offset,
                    "events": self.events[offset:offset + limit]}

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

    # -- run lifecycle ----------------------------------------------------------

    def start(self, cfg: dict, limit: int | None = None) -> dict:
        with self.lock:
            if self.status in ("running", "aborting"):
                raise HTTPException(409, "run already in progress")
            self.status = "running"
            self.events = []
            self.rows = []
            self.report = None
            self.dataset_desc = {}
            self.pipeline = pipeline_summary(cfg)
            self.run_seq += 1
            self.abort_event = threading.Event()
            self.t0 = time.monotonic()
        run_dir = alloc_run_dir(os.path.join(data_dir(), "runs"))
        with self.lock:
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
            ds = load_dataset(cfg.get("dataset_dir", ""),
                              cfg.get("dataset_file", "reads.jsonl"),
                              module_count=module_count,
                              verified_only=cfg.get("verified_only", True))
            if not ds.records:
                if ds.skipped_pending:
                    raise RuntimeError(
                        "no verified entries yet: "
                        f"{ds.skipped_pending} pending, 0 verified — curate "
                        "the set first (or untick 'Verified only')")
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
            workers = max(1, min(MAX_CONCURRENCY,
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
            summary = summarize(rows_now, width=module_count)
            finished = datetime.now(timezone.utc)
            times = [r["elapsed_s"] for r in rows_now
                     if r.get("elapsed_s") is not None]
            avg_time_s = round(sum(times) / len(times), 2) if times else None
            report = {
                "app": "ocr-vlm",
                "run": os.path.basename(run_dir),
                "pipeline": pipeline_summary(cfg),
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
                    "charset": cfg.get("charset", DEFAULT_CHARSET),
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
                    "classifier_backend": cfg.get("classifier_backend"),
                    "classifier_model": cfg.get("classifier_model") or None,
                    "classifier_min_conf": cfg.get(
                        "classifier_min_conf", 0.5),
                    "classifier_min_margin": cfg.get(
                        "classifier_min_margin", 0.1),
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
                    "low_conf_rows": sum(
                        1 for r in rows_now
                        if any(str(f).startswith("low-conf")
                               for f in (r.get("flags") or []))),
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
                f"{status}: VLM total {vlm['total']} mismatches "
                f"(avg {vlm['mean']}) vs baseline {saw['total']} "
                f"(avg {saw['mean']}) over {summary['scored_vlm']} scored rows; "
                f"{summary['errors']} error(s), {summary['adjusted']} adjusted"))
            with self.lock:
                self.report = report
                self.status = status
        except Exception as exc:
            self.log("error", f"run failed: {exc}")
            with self.lock:
                self.status = "failed"
                self.report = {"app": "ocr-vlm", "status": "failed",
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

    # -- per-row work -----------------------------------------------------------

    def _work(self, pos: int, rec: Record, ds, cfg: dict,
              module_count: int) -> dict | None:
        if self.abort_event.is_set():
            return None
        t0 = time.monotonic()
        reader = _reader_for(cfg)
        path = photo_path(ds.directory, rec.photo)
        if path is None:
            return self._error_row(pos, rec, "photo not found")
        img = cv2.imread(path)
        if img is None:
            return self._error_row(pos, rec, "photo unreadable (cv2.imread)")
        width = rec.width or module_count
        charset = cfg.get("charset", DEFAULT_CHARSET)
        try:
            # Blind read: expected="" — want is NEVER shown to the model or
            # the reconciler (the benchmark must not leak ground truth).
            # The reader prepares the photo itself: the tool path gets the
            # module annotation, the OCR text path the untouched photo.
            reading = reader.read(img, total=width, expected="",
                                  charset=charset, drum=charset)
        except ReaderError as exc:
            return self._error_row(pos, rec, f"reader failed: {exc}")
        scored = score_reading(reading.text, rec.want, rec.saw, width)
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
        low_conf = int(getattr(reader, "last_low_conf", 0) or 0)
        if low_conf:
            flags.append(f"low-conf:{low_conf}")
        segmentation = self._segmentation_meta(reader)
        composed = self._save_composed(pos, rec, cfg, reader)
        return {
            "index": pos,
            "photo": rec.photo,
            "prefix": prefix_of(rec.photo),
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
        """Per-row detection summary for the inspect card / report."""
        detected = getattr(reader, "last_detected", None)
        if detected is None:
            return None
        meta: dict = {"detected": bool(detected),
                      "display": getattr(reader, "last_display", None)}
        blanks = getattr(reader, "last_blanks", None)
        if blanks is not None:
            meta["blanks"] = int(sum(1 for b in blanks if b))
        return meta

    def _save_composed(self, pos: int, rec: Record, cfg: dict,
                       reader) -> str | None:
        """Write the composed image a detected read sent (debug artifact)."""
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

    def _error_row(self, pos: int, rec: Record, problem: str) -> dict:
        return {
            "index": pos, "photo": rec.photo, "prefix": prefix_of(rec.photo),
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
                f"(baseline {row['mm_saw']} mm)"
                + (" [text]" if row.get("read_mode") == "text" else "")))


def _sha256_of(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return ""


def _write_json(path: str, payload: dict) -> None:
    tmp = path + f".tmp-{os.getpid()}-{threading.get_ident()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(tmp, path)


harness = Harness()
app = FastAPI(title="Split-Flap OCR Benchmark")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as fh:
        return HTMLResponse(fh.read(), headers={"Cache-Control": "no-store"})


@app.get("/curate", response_class=HTMLResponse)
def curate_page():
    """Baseline-set curation UI (ground truth for the benchmark)."""
    with open(os.path.join(STATIC_DIR, "curate.html"),
              encoding="utf-8") as fh:
        return HTMLResponse(fh.read(), headers={"Cache-Control": "no-store"})


def _curate_call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except curate.CurateError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/config")
def get_config():
    return masked_config()


@app.post("/api/config")
def post_config(patch: dict):
    return save_config(patch)


@app.post("/api/dataset/validate")
def dataset_validate():
    cfg = load_config()
    if not cfg.get("dataset_dir"):
        raise HTTPException(400, "configure the dataset directory first")
    ds = load_dataset(cfg["dataset_dir"], cfg["dataset_file"],
                      module_count=cfg["module_count"],
                      verified_only=cfg["verified_only"])
    return describe(ds)


@app.get("/api/dataset/discover")
def dataset_discover(file: str = "reads.jsonl"):
    return {"dirs": discover(str(file or "reads.jsonl"))}


@app.get("/api/dataset/photo/{name}")
def dataset_photo(name: str):
    cfg = load_config()
    path = photo_path(cfg.get("dataset_dir", ""), name)
    if not valid_photo_name(name):
        raise HTTPException(400, "bad photo name")
    if path is None:
        raise HTTPException(404, "no such photo")
    media = "image/png"
    lowered = name.lower()
    if lowered.endswith((".jpg", ".jpeg")):
        media = "image/jpeg"
    elif lowered.endswith(".webp"):
        media = "image/webp"
    elif lowered.endswith(".bmp"):
        media = "image/bmp"
    return FileResponse(path, media_type=media)


@app.get("/api/dataset/preview/{name}")
def dataset_preview(name: str, mode: str | None = None,
                    style: str | None = None):
    """Render the configured segmentation for one photo (PNG).

    When detection fails the raw photo is returned instead and the
    ``X-Seg-Detected: 0`` header says so, so the UI always has an image
    to show and can compare it against the original.
    """
    cfg = load_config()
    if not valid_photo_name(name):
        raise HTTPException(400, "bad photo name")
    path = photo_path(cfg.get("dataset_dir", ""), name)
    if path is None:
        raise HTTPException(404, "no such photo")
    image = cv2.imread(path)
    if image is None:
        raise HTTPException(400, "photo unreadable (cv2.imread)")
    use_mode = str(mode or cfg.get("image_mode", "montage")).strip().lower()
    if use_mode not in segment.DETECTED_MODES or use_mode == "cells-detect":
        use_mode = "montage"  # the sheet is the visual for per-cell reads
    use_style = str(style or cfg.get("preprocess", "none")).strip().lower()
    if use_style not in segment.PREPROCESS_STYLES:
        use_style = "none"
    visual, meta = segment.segment_image(
        image, int(cfg.get("module_count", 12)), mode=use_mode,
        style=use_style)
    headers = {"Cache-Control": "no-store",
               "X-Seg-Detected": "0" if visual is None else "1"}
    if visual is None:
        visual = image
    else:
        headers["X-Seg-Blanks"] = str(meta.get("n_blank", ""))
    ok, buf = cv2.imencode(".png", visual)
    if not ok:
        raise HTTPException(500, "PNG encode failed")
    return Response(content=buf.tobytes(), media_type="image/png",
                    headers=headers)


@app.get("/api/baselines")
def baselines_list():
    """Every curated baseline set (ground truth) with its progress."""
    return {"root": curate.baselines_root(), "sets": curate.list_sets()}


@app.post("/api/baselines/create")
def baselines_create(body: dict | None = None):
    """Import a calib-vlm run (or reads.jsonl dir) into a new set."""
    body = body or {}
    module_count = int(load_config()["module_count"])
    return _curate_call(curate.create_set, str(body.get("source_dir") or ""),
                        name=(str(body.get("name")).strip()
                              if body.get("name") else None),
                        module_count=module_count)


@app.get("/api/baselines/{name}")
def baselines_get(name: str):
    return _curate_call(curate.get_set, name)


@app.post("/api/baselines/{name}/entry")
def baselines_entry(name: str, body: dict | None = None):
    """Save one entry's content (saving implies verified) or status."""
    body = body or {}
    photo = str(body.get("photo") or "")
    if not photo:
        raise HTTPException(400, "photo is required")
    content = body.get("content")
    status = body.get("status")
    entry = _curate_call(curate.update_entry, name, photo,
                         content=None if content is None else str(content),
                         status=status if status in ("pending", "verified")
                         else None)
    return {"ok": True, "entry": entry,
            "stats": curate.stats(curate.read_entries(name))}


@app.post("/api/baselines/{name}/remove-image")
def baselines_remove_image(name: str, body: dict | None = None):
    body = body or {}
    photo = str(body.get("photo") or "")
    if not photo:
        raise HTTPException(400, "photo is required")
    stats_out = _curate_call(curate.remove_image, name, photo)
    return {"ok": True, "stats": stats_out}


@app.delete("/api/baselines/{name}")
def baselines_delete(name: str):
    """Discard a whole baseline set (folder, entries and images)."""
    _curate_call(curate.delete_set, name)
    return {"ok": True}


@app.get("/api/baselines/{name}/photo/{photo}")
def baselines_photo(name: str, photo: str):
    if not valid_photo_name(photo):
        raise HTTPException(400, "bad photo name")
    path = curate.photo_file(name, photo)
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


@app.post("/api/run/start")
def run_start(body: dict | None = None):
    body = body or {}
    cfg = load_config()
    if not cfg.get("dataset_dir"):
        raise HTTPException(400, "configure the dataset directory first")
    if str(cfg.get("read_mode", "auto")) != "classify":
        for key in ("llm_base_url", "llm_model"):
            if not cfg.get(key):
                raise HTTPException(400, f"configure {key} first")
    _make_reader(cfg)  # validates provider/key (or the local model)
    limit = None
    raw_limit = body.get("limit")
    if raw_limit not in (None, ""):
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            raise HTTPException(400, "limit must be a positive integer")
        if limit <= 0:
            raise HTTPException(400, "limit must be a positive integer")
    return harness.start(cfg, limit)


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


@app.get("/api/run/rows")
def run_rows():
    return harness.rows_since()


@app.get("/api/run/row/{index}")
def run_row(index: int):
    row = harness.row_detail(index)
    if row is None:
        raise HTTPException(404, "no such row")
    return row


@app.get("/api/run/row/{index}/composed")
def run_row_composed(index: int):
    """The debug image a detected read sent for this row (PNG)."""
    row = harness.row_detail(index)
    if row is None or not row.get("composed"):
        raise HTTPException(404, "no composed image for this row")
    path = os.path.join(harness.run_dir, row["composed"])
    resolved = os.path.realpath(path)
    base = os.path.realpath(harness.run_dir)
    if not resolved.startswith(base + os.sep) or not os.path.isfile(resolved):
        raise HTTPException(404, "no composed image for this row")
    return FileResponse(resolved, media_type="image/png")


@app.get("/api/run/report")
def run_report():
    return harness.report or {}


def _avg_row_time(results_path: str) -> float | None:
    """Mean per-photo read time from a run's results.json (older runs)."""
    try:
        with open(results_path, encoding="utf-8") as fh:
            rows = json.load(fh).get("rows") or []
    except (OSError, ValueError):
        return None
    times = [r.get("elapsed_s") for r in rows
             if isinstance(r, dict) and r.get("elapsed_s") is not None]
    return round(sum(times) / len(times), 2) if times else None


@app.get("/api/runs")
def runs(limit: int = 25):
    """Past runs from disk (headline numbers for cross-run comparison)."""
    runs_dir = os.path.join(data_dir(), "runs")
    try:
        names = sorted(os.listdir(runs_dir), reverse=True)
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
        vlm = (summary.get("vlm") or {})
        saw = (summary.get("saw") or {})
        # Reports written before avg_time_s existed fall back to
        # results.json, so old runs are comparable too.
        avg_time_s = (rep.get("counts") or {}).get("avg_time_s")
        if avg_time_s is None:
            avg_time_s = _avg_row_time(
                os.path.join(runs_dir, name, "results.json"))
        # Reports written before the pipeline block existed rebuild it
        # from their config, so the Past-runs table stays comparable.
        pipeline_rec = rep.get("pipeline")
        pipeline = (pipeline_rec.get("text")
                    if isinstance(pipeline_rec, dict) else pipeline_rec)
        if not pipeline and isinstance(rep.get("config"), dict):
            pipeline = pipeline_summary(rep["config"])["text"]
        out.append({
            "run": rep.get("run") or name,
            "status": rep.get("status"),
            "finished": rep.get("finished"),
            "elapsed_s": rep.get("elapsed_s"),
            "llm_model": (rep.get("config") or {}).get("llm_model"),
            "pipeline": pipeline,
            "dataset_dir": (rep.get("dataset") or {}).get("dir"),
            "dataset_file": (rep.get("dataset") or {}).get("file"),
            "scored_vlm": summary.get("scored_vlm"),
            "vlm_total": vlm.get("total"),
            "vlm_avg": vlm.get("mean"),
            "saw_total": saw.get("total"),
            "saw_avg": saw.get("mean"),
            "avg_time_s": avg_time_s,
            "errors": summary.get("errors"),
        })
    return {"runs": out}


def main():
    import uvicorn

    uvicorn.run(app, host=os.environ.get("OCR_VLM_HOST", "127.0.0.1"),
                port=int(os.environ.get("OCR_VLM_PORT", "8003")),
                access_log=False)
