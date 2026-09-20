"""Filesystem layout for calib-auto.

Four data roots live inside the tool folder, deliberately, so the
git-tracked and git-ignored halves of the tool's state are obvious:

- ``golden/``    labeled photos + verified content — the ground truth
                  for both training and benchmarking (TRACKED by git)
- ``models/``     the shipped deployment artifact (``cnn.pt`` plus its
                  ``cnn.json`` sidecar) — ready to use with no training
                  (TRACKED by git)
- ``training/``  derived glyph caches and rebuildable models
                  (NOT tracked; regenerated from ``golden/``)
- ``runs/``      every calibration run, benchmark run, training job
                  log and read-test capture (NOT tracked; gitignored)

``CALIB_AUTO_DATA`` overrides the base folder (tests use it for
isolation; a deployment could point it at a bigger disk).
"""

from __future__ import annotations

import os

GOLDEN_DIRNAME = "golden"
MODELS_DIRNAME = "models"
TRAINING_DIRNAME = "training"
RUNS_DIRNAME = "runs"


def tool_root() -> str:
    """The tool folder (parent of the package directory)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def base_root() -> str:
    """Base folder for all data roots (env override for tests)."""
    return os.environ.get("CALIB_AUTO_DATA") or tool_root()


def golden_root() -> str:
    return os.path.join(base_root(), GOLDEN_DIRNAME)


def models_root() -> str:
    """Home of the shipped deployment artifact (tracked by git)."""
    return os.path.join(base_root(), MODELS_DIRNAME)


def training_root() -> str:
    return os.path.join(base_root(), TRAINING_DIRNAME)


def runs_root() -> str:
    return os.path.join(base_root(), RUNS_DIRNAME)


def config_path() -> str:
    return os.path.join(base_root(), "config.json")


def alloc_run_dir(prefix: str = "run") -> str:
    """Allocate the first unused ``runs/<prefix>-NNN`` dir.

    Collision-proof (creates atomically): calibration runs use the
    ``cal`` prefix, benchmark runs ``bench``, training jobs ``train``.
    """
    runs = runs_root()
    os.makedirs(runs, exist_ok=True)
    n = 1
    while True:
        run_dir = os.path.join(runs, f"{prefix}-{n:03d}")
        try:
            os.makedirs(run_dir, exist_ok=False)
            return run_dir
        except FileExistsError:
            n += 1
