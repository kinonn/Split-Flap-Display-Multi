"""CNN read path: one display photo -> per-module reading, fully local.

The CNN approach's recognizer. It implements the same contract as
``VlmReader`` (see ``reader.py``) so the calibration loop is identical
for both approaches:

- ``annotate = False`` — the CNN gets the raw photo (no index ticks; it
  uses the OpenCV display detector instead),
- ``read(jpeg_bytes, total, expected, charset, drum) -> Reading`` —
  decode, locate the display, split modules, optionally blank-gate each
  cell (``blank_gate`` opt-in), and classify the rest with the trained
  model (template bank or CNN),
- ``error_reading`` / ``last_calls`` / ``last_usage`` — same shapes; a
  local read is one "call" and zero tokens,
- per-cell confidence and margin ride along in the ModuleReadings, and
  cells under the configured floors are counted and named in the row
  warnings: a weak read is *visible*, never silently trusted.

No provider, no network, no API key.
"""

from __future__ import annotations

import hashlib
import json
import os

import cv2
import numpy as np

from . import classifier, segment, train_cnn
from .reader import ModuleReading, Reading

BACKENDS = ("auto", "bank", "cnn")


def resolve_model(backend: str = "auto",
                  explicit: str | None = None) -> tuple[str, str]:
    """(backend, model path) — ``auto`` prefers the CNN artifact.

    The CNN is the intended deployment backend; the bank stays as the
    dependency-free fallback and for A/B evaluation. ``explicit``
    overrides both (its name decides the loader only by extension —
    ``.pt`` is CNN, anything else is a bank npz).
    """
    backend = str(backend or "auto").lower()
    if backend not in BACKENDS:
        backend = "auto"
    if explicit:
        chosen = "cnn" if str(explicit).lower().endswith(".pt") else "bank"
        return chosen, str(explicit)
    cnn_path = train_cnn.model_path()
    bank_path = classifier.bank_path()
    if backend in ("auto", "cnn") and os.path.isfile(cnn_path):
        return "cnn", cnn_path
    if backend == "cnn":
        return "cnn", cnn_path  # load will raise a pointed BankError
    return "bank", bank_path


def _sha256(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return ""


def pipeline_info(backend: str = "auto", model_path: str | None = None,
                  min_conf: float = 0.5, min_margin: float = 0.1,
                  blank_gate: bool = False) -> dict:
    """Provenance record for a CNN run's pipeline block.

    Pure file I/O (no model load): the artifact is hashed and its
    sidecar meta read, so a report names exactly which model answered —
    and a moved/retrained artifact is detectable from the hash.
    """
    resolved_backend, path = resolve_model(backend, model_path)
    info = {
        "backend": resolved_backend,
        "path": path,
        "min_conf": float(min_conf),
        "min_margin": float(min_margin),
        "blank_gate": bool(blank_gate),
        "sha256": "",
    }
    meta_file = os.path.splitext(path)[0] + ".json"
    try:
        with open(meta_file, encoding="utf-8") as fh:
            meta = json.load(fh)
        if isinstance(meta, dict):
            for key in ("sets", "trained", "created"):
                if meta.get(key) is not None:
                    info[key] = meta[key]
            classes = meta.get("classes")
            if isinstance(classes, list):
                info["classes"] = len(classes)
    except (OSError, ValueError):
        pass
    info["sha256"] = _sha256(path) if os.path.isfile(path) else ""
    return info


class CnnReader:
    """One photo (JPEG bytes) -> Reading via the trained local model."""

    def __init__(self, backend: str = "auto", model_path: str | None = None,
                 min_conf: float = 0.5, min_margin: float = 0.1,
                 blank_gate: bool = False,
                 model=None):
        resolved_backend, resolved_path = resolve_model(backend, model_path)
        self.backend = resolved_backend
        self.model_path = resolved_path
        self.min_conf = float(min_conf)
        self.min_margin = float(min_margin)
        self.blank_gate = bool(blank_gate)
        self._model = model  # injectable for tests
        # The calibration loop reads these (same shapes as VlmReader).
        self.annotate = False
        self.vlm = None
        self.last_calls = 1
        self.last_usage: dict = {}
        self.last_low_conf = 0
        # Detection metadata (kept for reports and debugging).
        self.last_no_detect = False
        self.last_detected: bool | None = None
        self.last_display: dict | None = None
        self.last_blanks: list[bool] | None = None
        self.last_boxes: list[tuple[int, int, int, int]] | None = None
        self.last_image_size: tuple[int, int] | None = None
        self.last_grid_support = 1.0
        self.last_composed = None

    @property
    def model(self):
        if self._model is None:
            if self.backend == "cnn":
                self._model = train_cnn.CnnClassifier.load(self.model_path)
            else:
                self._model = classifier.GlyphBank.load(self.model_path)
        return self._model

    def info(self) -> dict:
        """Model provenance (best effort; never raises)."""
        try:
            model = self.model
        except classifier.BankError:
            return {"backend": self.backend, "path": self.model_path,
                    "error": "model not loadable"}
        meta = getattr(model, "meta", {}) or {}
        info = {
            "backend": self.backend,
            "path": self.model_path,
            "classes": len(getattr(model, "classes", [])),
            "sets": meta.get("sets"),
            "trained": meta.get("trained"),
            "created": meta.get("created"),
            "min_conf": self.min_conf,
            "min_margin": self.min_margin,
            "blank_gate": self.blank_gate,
            "device": str(getattr(model, "device", "cpu")),
        }
        return info

    def read(self, data, total: int, expected: str = "",
             charset: str = "", drum: str = "") -> Reading:
        """``data`` is JPEG bytes (what the loop sends) or a BGR array."""
        if isinstance(data, (bytes, bytearray, memoryview)):
            image = cv2.imdecode(np.frombuffer(bytes(data), np.uint8),
                                 cv2.IMREAD_COLOR)
        else:
            image = data
        if image is None:
            raise ValueError("photo could not be decoded")
        self.last_image_size = (int(image.shape[1]), int(image.shape[0]))
        self.last_no_detect = False
        self.last_detected = None
        self.last_display = None
        self.last_blanks = None
        self.last_boxes = None
        self.last_grid_support = 1.0
        self.last_composed = None
        self.last_low_conf = 0
        self.last_calls = 1
        self.last_usage = {}

        total = max(1, int(total))
        display = segment.find_display(image)
        if display is None:
            self.last_no_detect = True
            self.last_detected = False
            note = ("display not detected — check camera framing/lighting "
                    "(classifier mode has no strip fallback)")
            reading = self.error_reading(total, expected, note)
            reading.unreliable = True
            reading.unreliable_note = note
            return reading
        boxes = segment.module_boxes(image, display, total)
        crops = segment.crop_modules(image, boxes)
        blanks = ([segment.is_blank(crop) for crop in crops]
                  if self.blank_gate else [False] * len(crops))
        self.last_detected = True
        self.last_display = display.as_dict()
        self.last_blanks = blanks
        self.last_boxes = [tuple(int(v) for v in box) for box in boxes]
        # Does the grid actually sit on modules? A detection clipped by a
        # washed-out end of the flap band leaves fewer columns than
        # modules; the fit recovers most of them, but a grid that still
        # does not explain the seams cannot be trusted — its cells are not
        # on modules, so every character below would be a guess dressed up
        # as a reading.
        support = segment.grid_support(image, boxes)
        self.last_grid_support = support
        misaligned = support < segment.GRID_MIN_SUPPORT

        model = self.model
        modules: list[ModuleReading] = []
        low_conf = 0
        for i, crop in enumerate(crops):
            char, conf, margin, source = classifier.classify_cell(
                model, crop, blank=blanks[i], blank_gate=self.blank_gate)
            if (source == "classifier" and char != " "
                    and (conf < self.min_conf or margin < self.min_margin)):
                low_conf += 1
            if misaligned:
                # Keep the character (the report should still show what the
                # model saw) but drop the confidence and the condition, so
                # the loop neither tunes on it nor calls it clean.
                conf = 0.0
            modules.append(ModuleReading(
                i, char, "unreadable" if misaligned
                else ("blank" if char == " " else "clean"),
                round(min(1.0, max(0.0, conf)), 3), source,
                expected[i] if i < len(expected) else " "))

        warnings: list[str] = []
        blank_n = sum(1 for b in blanks if b)
        if blank_n and self.blank_gate:
            warnings.append(f"{blank_n}/{total} blank cells decided by "
                            "OpenCV (no model call)")
        if low_conf:
            warnings.append(
                f"{low_conf} low-confidence cell(s) below conf "
                f"{self.min_conf} or margin {self.min_margin}")
        note = ""
        if misaligned:
            note = (f"module grid does not fit the display "
                    f"({support:.0%} of module gaps found) — cells are not "
                    f"reliably on their modules, check camera "
                    f"framing/lighting")
            warnings.append(note)
        self.last_low_conf = low_conf
        self.last_composed = segment.montage(crops, total, style="none")
        return Reading(modules, realigned=False,
                       raw_count=sum(1 for m in modules if m.char != " "),
                       warnings=warnings, unreliable=misaligned,
                       unreliable_note=note)

    def error_reading(self, total: int, expected: str,
                      problem: str) -> Reading:
        """All-unreadable stand-in (same shape as the VLM reader's)."""
        modules = [ModuleReading(i, "?", "unreadable", 0.0, "error",
                                 expected[i] if i < len(expected) else " ")
                   for i in range(total)]
        return Reading(modules, realigned=False, raw_count=0,
                       warnings=[problem])
