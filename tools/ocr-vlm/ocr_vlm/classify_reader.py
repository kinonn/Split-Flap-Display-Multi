"""Local-classifier read path (``read_mode="classify"``).

No provider, no network: the display is located with the same OpenCV
detector the other detected modes use, each module cell is blank-gated,
and non-blank cells are classified by the trained glyph model — the
template bank or the CNN (both expose ``predict_raster``). Per-cell
confidence and margin ride along in the ``ModuleReading``s, and cells
under the configured floor are counted and named in the row warnings: a
weak read is *visible*, never silently trusted.

Read state mirrors ``ModeReader`` (``last_mode``, ``last_raw``,
``last_no_detect``, ``last_display``, ``last_blanks``, ``last_composed``
…) so the server's row builder, composed-image debug saving and
pipeline capture work unchanged. ``vlm`` is ``None`` on purpose — the
server already treats usage as optional.
"""

from __future__ import annotations

import hashlib
import json
import os

from calib_vlm.reader import ModuleReading, Reading

from . import classifier, glyphs, segment, train_cnn

BACKENDS = ("auto", "bank", "cnn")


def resolve_model(data_dir: str, backend: str = "auto",
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
    folder = glyphs.cache_dir(data_dir)
    cnn_path = train_cnn.model_path(data_dir)
    bank_path = classifier.bank_path(data_dir)

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


def pipeline_info(cfg: dict, data_dir_path: str) -> dict:
    """Provenance record for a classify run's pipeline block.

    Pure file I/O (no model load): the artifact is hashed and its
    sidecar meta read, so a report names exactly which model answered —
    and a moved/retrained artifact is detectable from the hash.
    """
    backend = str(cfg.get("classifier_backend", "auto"))
    resolved_backend, path = resolve_model(
        data_dir_path, backend, cfg.get("classifier_model") or None)
    info = {
        "backend": resolved_backend,
        "path": path,
        "min_conf": float(cfg.get("classifier_min_conf", 0.5)),
        "min_margin": float(cfg.get("classifier_min_margin", 0.1)),
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


class ClassifyReader:
    """One photo -> Reading via the trained local glyph model."""

    def __init__(self, backend: str = "auto", model_path: str | None = None,
                 data_dir: str = "", min_conf: float = 0.5,
                 min_margin: float = 0.10, model=None):
        resolved_backend, resolved_path = resolve_model(
            data_dir, backend, model_path)
        self.backend = resolved_backend
        self.model_path = resolved_path
        self.min_conf = float(min_conf)
        self.min_margin = float(min_margin)
        self._model = model  # injectable for tests
        # Row metadata (same names the server reads after every read()).
        self.last_mode = "classify"
        self.last_fallback = False
        self.last_empty = False
        self.last_raw: str | None = None
        self.last_no_detect = False
        self.last_detected: bool | None = None
        self.last_display: dict | None = None
        self.last_blanks: list[bool] | None = None
        self.last_composed = None
        self.last_low_conf = 0
        self.vlm = None

    @property
    def model(self):
        if self._model is None:
            if self.backend == "cnn":
                self._model = train_cnn.CnnClassifier.load(self.model_path)
            else:
                self._model = classifier.GlyphBank.load(self.model_path)
        return self._model

    def info(self) -> dict:
        """Model provenance for the pipeline record (best effort)."""
        try:
            model = self.model
        except classifier.BankError:
            return {"backend": self.backend, "path": self.model_path,
                    "error": "model not loadable"}
        meta = getattr(model, "meta", {}) or {}
        return {
            "backend": self.backend,
            "path": self.model_path,
            "classes": len(getattr(model, "classes", [])),
            "sets": meta.get("sets"),
            "trained": meta.get("trained"),
            "created": meta.get("created"),
            "min_conf": self.min_conf,
            "min_margin": self.min_margin,
        }

    def read(self, image, total: int, expected: str = "",
             charset: str = "", drum: str = "") -> Reading:
        self.last_fallback = False
        self.last_empty = False
        self.last_raw = None
        self.last_no_detect = False
        self.last_detected = None
        self.last_display = None
        self.last_blanks = None
        self.last_composed = None
        self.last_low_conf = 0
        if self.backend == "cnn":
            self.model  # load eagerly so a missing artifact fails the read

        total = max(1, int(total))
        display = segment.find_display(image)
        if display is None:
            self.last_no_detect = True
            self.last_detected = False
            modules = [
                ModuleReading(i, " ", "blank", 1.0, "cv",
                              expected[i] if i < len(expected) else " ")
                for i in range(total)]
            return Reading(
                modules, raw_count=0,
                warnings=["display not detected; classifier mode has no "
                          "strip fallback (read as all blanks)"])
        boxes = segment.module_boxes(image, display, total)
        crops = segment.crop_modules(image, boxes)
        blanks = [segment.is_blank(crop) for crop in crops]
        self.last_detected = True
        self.last_display = display.as_dict()
        self.last_blanks = blanks

        model = self.model
        modules: list[ModuleReading] = []
        weak = 0
        for i, crop in enumerate(crops):
            char, conf, margin, source = classifier.classify_cell(
                model, crop, blank=blanks[i])
            if (source == "classifier" and char != " "
                    and (conf < self.min_conf or margin < self.min_margin)):
                weak += 1
            modules.append(ModuleReading(
                i, char, "blank" if char == " " else "clean",
                round(min(1.0, max(0.0, conf)), 3), source,
                expected[i] if i < len(expected) else " "))

        warnings: list[str] = []
        blank_n = sum(1 for b in blanks if b)
        if blank_n:
            warnings.append(f"{blank_n}/{total} blank cells decided by "
                            "OpenCV (no model call)")
        if weak:
            warnings.append(
                f"{weak} low-confidence cell(s) below conf "
                f"{self.min_conf} or margin {self.min_margin}")
        self.last_low_conf = weak
        self.last_composed = segment.montage(crops, total, style="none")
        shown = " ".join(f"{m.char}:{m.confidence:.2f}" for m in modules
                         if m.source == "classifier")
        self.last_raw = shown[:500] or None
        self.last_empty = all(m.char == " " for m in modules)
        return Reading(modules, realigned=False,
                       raw_count=sum(1 for m in modules if m.char != " "),
                       warnings=warnings)
