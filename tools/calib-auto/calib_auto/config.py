"""Unified configuration for calib-auto.

One flat config (``config.json`` next to the data roots, mode 0600,
server-side only) covers all four surfaces of the tool: the shared
display/camera setup, the calibration loop options, the VLM provider,
and the benchmark/classifier knobs. ``load_config()`` applies defaults
and clamps; ``save_config(patch)`` validates a partial update and
merges it; ``masked_config()`` is the API-safe view (the API key never
leaves the server unmasked).

Validation raises ``ConfigError`` (a ValueError) so this module works
without FastAPI; the server converts it to HTTP 400.
"""

from __future__ import annotations

import json
import math
import os

from . import paths

# Charset 48 of the drum contract (src/web/calib-contract.json).
DEFAULT_CHARSET = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':?!.-/$@#%"
DEFAULT_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_MODEL = "deepseek-v4-flash-vision-exp"
DEFAULT_OCR_PROMPT = "OCR:"
MAX_CONCURRENCY = 8

READ_MODES = ("auto", "tool", "text", "classify")
IMAGE_MODES = ("strip", "cells", "montage", "cells-detect", "strip-detect")
PREPROCESS_STYLES = ("none", "contrast", "invert", "binary")
IMAGE_FORMATS = ("jpeg", "png")
CLASSIFIER_BACKENDS = ("auto", "bank", "cnn")
RUN_MODES = ("dry-run", "full")

# Numeric keys: (default, lo, hi) — every one clamped on load and save.
_NUMERIC = {
    "module_count": (12, 1, 64),
    "camera_index": (0, 0, 16),
    "camera_width": (1280, 160, 4096),
    "camera_height": (720, 120, 4096),
    "brightness": (50.0, 0.0, 100.0),
    "crop_percent": (0.0, 0.0, 40.0),
    "dwell_ms": (800, 0, 60000),
    "timeout_s": (60.0, 1.0, 600.0),
    "max_seconds": (3600.0, 60.0, 86400.0),
    "min_confidence": (0.6, 0.0, 1.0),
    "image_max_width": (1024, 256, 4096),
    "image_quality": (80, 1, 100),
    "ocr_max_tokens": (128, 0, 4096),
    "concurrency": (1, 1, MAX_CONCURRENCY),
    "classifier_min_conf": (0.5, 0.0, 1.0),
    "classifier_min_margin": (0.1, 0.0, 1.0),
}
# Optional numeric keys: default None; a value clamps into its range.
_OPTIONAL_NUMERIC = {
    "exposure": (-64.0, 64.0),
    "warmup_s": (0.0, 30.0),
}


class ConfigError(ValueError):
    """Invalid configuration value (the server maps it to HTTP 400)."""


def _int_or(value, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def _float_or(value, default: float, lo: float, hi: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return max(lo, min(hi, out))


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


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def load_config() -> dict:
    """Current config: file + env overrides + defaults + clamps."""
    cfg: dict = {}
    try:
        with open(paths.config_path(), encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        pass
    # Retired keys, see apply_patch.
    cfg.pop("skip_chars", None)
    cfg.pop("skip_enabled", None)
    env_map = {"display_host": "CALIB_AUTO_DISPLAY_HOST",
               "llm_base_url": "LLM_BASE_URL", "llm_model": "LLM_MODEL",
               "llm_api_key": "LLM_API_KEY"}
    for key, env in env_map.items():
        value = _env(env)
        if value:
            cfg[key] = value

    cfg.setdefault("display_host", "splitflap.local")
    cfg.setdefault("dataset_dir", "")
    cfg.setdefault("dataset_file", "labels.jsonl")
    cfg.setdefault("charset", DEFAULT_CHARSET)
    if not str(cfg.get("charset") or ""):
        cfg["charset"] = DEFAULT_CHARSET
    for key, (default, lo, hi) in _NUMERIC.items():
        cfg[key] = _int_or(cfg.get(key), default, lo, hi) if isinstance(
            default, int) else _float_or(cfg.get(key), default, lo, hi)
    for key, (lo, hi) in _OPTIONAL_NUMERIC.items():
        value = cfg.get(key)
        cfg[key] = None if value in (None, "") else _float_or(value, 0.0,
                                                              lo, hi)
    cfg["mode"] = (str(cfg.get("mode") or "full").strip().lower()
                   if str(cfg.get("mode") or "full").strip().lower()
                   in RUN_MODES else "full")
    cfg["read_mode"] = (str(cfg.get("read_mode") or "auto").strip().lower()
                        if str(cfg.get("read_mode") or "auto").strip().lower()
                        in READ_MODES else "auto")
    mode = str(cfg.get("image_mode") or "strip").strip().lower()
    cfg["image_mode"] = mode if mode in IMAGE_MODES else "strip"
    style = str(cfg.get("preprocess") or "none").strip().lower()
    cfg["preprocess"] = style if style in PREPROCESS_STYLES else "none"
    fmt = str(cfg.get("image_format") or "jpeg").strip().lower()
    cfg["image_format"] = fmt if fmt in IMAGE_FORMATS else "jpeg"
    backend = str(cfg.get("classifier_backend") or "auto").strip().lower()
    cfg["classifier_backend"] = (backend if backend in CLASSIFIER_BACKENDS
                                 else "auto")
    cfg["classifier_model"] = str(cfg.get("classifier_model") or "").strip()
    cfg.setdefault("ocr_prompt", DEFAULT_OCR_PROMPT)
    if not str(cfg.get("ocr_prompt") or ""):
        cfg["ocr_prompt"] = DEFAULT_OCR_PROMPT
    cfg["blank_gate"] = _bool_or(cfg.get("blank_gate"), False)
    cfg["exhaustive"] = _bool_or(cfg.get("exhaustive"), False)
    cfg["debug_images"] = _bool_or(cfg.get("debug_images"), False)
    cfg["verified_only"] = _bool_or(cfg.get("verified_only"), True)
    cfg.setdefault("llm_base_url", DEFAULT_BASE_URL)
    cfg.setdefault("llm_model", DEFAULT_MODEL)
    cfg.setdefault("llm_api_key", "")
    return cfg


def save_config(patch: dict) -> dict:
    """Validate + merge a partial update; returns the masked config."""
    os.makedirs(paths.base_root(), exist_ok=True)
    stored: dict = {}
    try:
        with open(paths.config_path(), encoding="utf-8") as fh:
            stored = json.load(fh)
    except (OSError, ValueError):
        pass
    # Retired keys (character exclusions were removed; every drum
    # character is always calibrated): drop them so the file self-cleans.
    stored.pop("skip_chars", None)
    stored.pop("skip_enabled", None)

    for key in ("display_host", "llm_base_url", "llm_model",
                "dataset_dir", "dataset_file"):
        if key in patch and patch[key] is not None:
            value = str(patch[key]).strip()
            # dataset_dir is deliberately settable to "" (unconfigured).
            if key == "dataset_dir" and not value:
                stored[key] = ""
                continue
            if not value:
                raise ConfigError(f"{key} must not be empty")
            stored[key] = value
    if "charset" in patch and patch["charset"] is not None:
        value = str(patch["charset"])
        if not value:
            raise ConfigError("charset must not be empty")
        stored["charset"] = value
    if "ocr_prompt" in patch and patch["ocr_prompt"] is not None:
        stored["ocr_prompt"] = (str(patch["ocr_prompt"]).strip()
                                or DEFAULT_OCR_PROMPT)
    if "mode" in patch and patch["mode"] not in (None, ""):
        value = str(patch["mode"]).strip().lower()
        if value not in RUN_MODES:
            raise ConfigError("mode must be dry-run or full")
        stored["mode"] = value
    if "read_mode" in patch and patch["read_mode"] not in (None, ""):
        value = str(patch["read_mode"]).strip().lower()
        if value not in READ_MODES:
            raise ConfigError(
                "read_mode must be auto, tool, text or classify")
        stored["read_mode"] = value
    if "image_mode" in patch and patch["image_mode"] not in (None, ""):
        value = str(patch["image_mode"]).strip().lower()
        if value not in IMAGE_MODES:
            raise ConfigError("image_mode must be one of "
                              + ", ".join(IMAGE_MODES))
        stored["image_mode"] = value
    if "preprocess" in patch and patch["preprocess"] not in (None, ""):
        value = str(patch["preprocess"]).strip().lower()
        if value not in PREPROCESS_STYLES:
            raise ConfigError("preprocess must be one of "
                              + ", ".join(PREPROCESS_STYLES))
        stored["preprocess"] = value
    if "image_format" in patch and patch["image_format"] not in (None, ""):
        value = str(patch["image_format"]).strip().lower()
        if value not in IMAGE_FORMATS:
            raise ConfigError("image_format must be jpeg or png")
        stored["image_format"] = value
    if "classifier_backend" in patch and patch["classifier_backend"] \
            not in (None, ""):
        value = str(patch["classifier_backend"]).strip().lower()
        if value not in CLASSIFIER_BACKENDS:
            raise ConfigError("classifier_backend must be auto, bank or cnn")
        stored["classifier_backend"] = value
    if "classifier_model" in patch and patch["classifier_model"] is not None:
        stored["classifier_model"] = str(patch["classifier_model"]).strip()
    for key in ("exhaustive", "debug_images",
                "verified_only"):
        if key in patch and patch[key] is not None:
            stored[key] = _bool_or(patch[key], True)
    if "blank_gate" in patch and patch["blank_gate"] is not None:
        stored["blank_gate"] = _bool_or(patch["blank_gate"], False)
    for key, (default, lo, hi) in _NUMERIC.items():
        if key in patch and patch[key] not in (None, ""):
            try:
                value = float(patch[key])
            except (TypeError, ValueError):
                raise ConfigError(f"{key} must be a number")
            if not math.isfinite(value):
                raise ConfigError(f"{key} must be a number")
            if not lo <= value <= hi:
                raise ConfigError(
                    f"{key} must be between {lo:g} and {hi:g}")
            stored[key] = int(value) if isinstance(default, int) else value
    for key, (lo, hi) in _OPTIONAL_NUMERIC.items():
        if key in patch:
            if patch[key] in (None, ""):
                stored[key] = None
            else:
                try:
                    value = float(patch[key])
                except (TypeError, ValueError):
                    raise ConfigError(f"{key} must be a number or null")
                if not math.isfinite(value):
                    raise ConfigError(f"{key} must be a number or null")
                if not lo <= value <= hi:
                    raise ConfigError(
                        f"{key} must be between {lo:g} and {hi:g} or null")
                stored[key] = value
    if patch.get("llm_api_key"):
        stored["llm_api_key"] = str(patch["llm_api_key"])
    _atomic_write_json(paths.config_path(), stored)
    return masked_config()


def _atomic_write_json(path: str, payload: dict) -> None:
    """Write JSON with mode 0600, atomically (temp file + os.replace)."""
    tmp = path + ".tmp"
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
