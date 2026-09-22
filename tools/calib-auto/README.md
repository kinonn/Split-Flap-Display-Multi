# calib-auto

Unified, standalone auto-calibration tool for the modular split-flap
display. One server, one web UI, two recognizer approaches — plus the
supporting tools to curate data, train the local model, and pick the
best vision model:

| Surface | Page | What it does |
| --- | --- | --- |
| **CNN approach** | `/cnn` | Calibrate with the local trained model — no provider, no network, fast |
| **VLM approach** | `/vlm` | Calibrate with a vision-language model through the `report_reading` tool call |
| **Golden set** | `/golden` | Curate labeled photos (the ground truth for training and testing) |
| **Train CNN** | `/train` | Build glyph caches and train/retrain the local model, live |
| **Test CNN** | `/cnn-test` | Read golden photos with the local model — single reads and full benchmark runs scored against the curated content |
| **Test VLMs** | `/bench` | Benchmark any vision model against the golden set |
| **Review runs** | `/cal-review` | Review completed calibration captures with CNN module-box overlays and per-character confidence/source cards |

Both calibration approaches run the **same loop** — phases, budgets,
safety rails, fleet handling, persistence are identical — only the
recognizer differs, so results are comparable across approaches.

## Quick start

```sh
cd tools/calib-auto
uv sync
uv run calib-auto-server        # → http://127.0.0.1:8004
```

1. **Curate a golden set** (`/golden`): import a calibration run or an
   old tool's set, then verify each photo's content. Only verified
   entries are used for training and benchmarking.
2. **Train the CNN** (`/train`): builds glyph caches from the golden
   set, validates on a photo-disjoint split, then trains the deployment
   artifact. Uses CUDA when available, else MPS when available, else CPU
   (a few minutes on CPU).

GPU note: `uv sync` installs CUDA-enabled torch on Linux/Windows via
the PyTorch cu126 index (see `pyproject.toml`; macOS falls back to a
CPU build). Verify with
`uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` —
training logs `device: cuda` when the GPU is picked up. Force a
backend with the `/train` device dropdown, `--device`, or
`CALIB_AUTO_DEVICE`.
3. **Calibrate** (`/cnn` or `/vlm`): set the display host and camera,
   run a dry-run first (read-only shift table), then a full run.
   Every verified fix is persisted; the pre-run settings snapshot stays
   on disk for manual rollback (`Restore snapshot` in the UI).

Both calibration pages show a **live camera view** (auto-refreshing
frame, ~1 fps), off by default — tick *Live camera view* to start it.
Next to it is a **sensor exposure** control: leave *Auto exposure*
checked for driver auto-exposure, or uncheck it and drag the slider to
fix a manual value. The view pauses automatically while a run holds the
camera and resumes when the run ends.

## The calibration loop

Phases (independently selectable; P0 always runs as a read-only gate):

- **P0 register** — blank, all-H and index-strip frames must read back
  (camera framing/focus gate; mirrored reads rejected).
- **P1 coarse** — reverse uniform sweep over the whole drum; each
  module's shift histogram is aggregated per direction (dominant
  direction wins, magnitude = smallest gap) and that direction's share
  of the drum picks the module-cell correction: >= 75% earns full
  pitch multiples, 12.5–75% a proportional correction, below that no
  module move with the gaps left as per-character P2 work. Only a
  starved histogram escalates instead of being "fixed".
- **P2 fine** — per-character offsets from the P1 residual map, tuned
  with a parallel coarse-to-fine ladder (one cell per module per wave,
  firmware ±32 char-cell clamp respected, no-op candidates rejected).
- **P4 verify** — repeatability plus short forward boundary hops.
- **acceptance** — forward sweep; every module must read every
  non-space character clean. Confusable pairs (O/0, I/1, S/5, Z/2,
  B/8, G/6) are no-information and never judged.

Safety rails: motor-wear/time budgets (frames, reader calls, previews,
persists, sweeps), unreliable-read escalation (abort when the reader
cannot be trusted), dry-run mode, and a pre-run settings snapshot. A *reader* that cannot
trust its own reading — display not located, module grid not on the
modules (CNN), photo unparsable (VLM) — says so, and three such readings
in a row abort the run with that note, so bad framing/lighting surfaces in
seconds instead of after a full sweep.

**Fleet support**: master + ESP-NOW remotes. Group 1 (local) uses
preview-then-persist; remote groups (no remote preview endpoint in
firmware) use persist-verify-revert per cell. Group widths come from
the firmware's `groupWidths` / `masterGroupModuleCounts`; inconsistent
geometry is an error, never a guess.

## The recognizers

Both implement the same contract (`read(jpeg_bytes, total, expected,
charset, drum) -> Reading`), so the loop is agnostic:

- **CNN** (`cnn_reader.py` + `segment.py`/`classifier.py`/`train_cnn.py`)
  — locate the display with OpenCV, split the module grid, blank-gate
  each cell, classify the rest with the trained model (or the
  dependency-free template bank). The split is refitted to the display's
  module seams (`segment._fit_bounds`): the detected dark run can be
  clipped by a lit end of the flap band, and an equal-pitch split of a
  clipped box mis-places every cell. A grid that still cannot be shown to
  sit on the modules makes the reading *unreliable* (characters kept for
  the report, confidences dropped) instead of feeding plausible-looking
  garbage to the loop. Per-cell confidence rides along; cells under the
  floors are counted and flagged. The VLM-style photo annotation is
  skipped (the CNN uses the detector instead).
- **VLM** (`reader.py` + `vlm.py` + `prompts.py`) — the photo gets the
  module annotation (separators + index ticks, production parity) and
  goes to the configured OpenAI-compatible provider with the
  `report_reading` tool schema. The reader reconciles the answer against
  the module count (blanks included) and flags anything it had to infer.
  Thinking-mode providers that answer in prose JSON are handled.

## Golden set, training data, runs

Four data roots live inside the tool folder:

```
golden/     labeled photos + verified content   (TRACKED by git)
models/     shipped CNN artifact, ready to use  (TRACKED by git)
training/   glyph caches + rebuildable models   (NOT tracked)
runs/       every run/job artifact              (NOT tracked)
```

Calibration run reports now persist CNN `frames[].boxes` and the detected
display rectangle alongside each frame's module readings. Open `/cal-review`
to select a completed `cal-NNN` run, move through its captured images, see
the module boxes overlaid, and inspect each detected character's confidence
and source (`Model` or `CV2`). Reports created before boxes were persisted
are supported too: the review endpoint recreates their geometry from the
saved image using the same JPEG and segmentation path.

- `golden/<set>/{set.json, labels.jsonl, images/}` — a set is imported
  from an old tool's run (`report.json`), a curated set
  (`labels.jsonl`/`baseline.jsonl`, verified status preserved) or a
  `reads.jsonl` dataset. Images are recompressed on import (PNG level 9;
  JPEG q92 when a set's PNG total exceeds the budget) to keep the
  tracked set a sane git size. The original source path is not stored in
  `set.json`.
- `training/` — `<set>.npz` glyph caches (+ JSON summaries with
  per-class counts and skip reasons) and `bank.npz` (template bank).
  Regenerated from `golden/` via the `/train` page or the CLI below —
  never committed.
- `models/` — `cnn.pt` (CNN, with `cnn.json` metrics + provenance),
  the deployment artifact. Committed so a fresh clone can calibrate
  with no training; retraining overwrites it in place.
- `runs/cal-NNN`, `runs/bench-NNN`, `runs/train-NNN` — photos, events,
  reports and snapshots per run.

Env: `CALIB_AUTO_DATA` overrides the base folder; `CALIB_AUTO_HOST` /
`CALIB_AUTO_PORT` the listen address (default `127.0.0.1:8004`);
`CALIB_AUTO_LOG_LEVEL` (default `warning`) and `CALIB_AUTO_ACCESS_LOG`
(default off — the UI polls the live view continuously).

## CLI equivalents

Every UI surface has a CLI (all paths relative to `tools/calib-auto`):

```sh
uv run python -m calib_auto.golden list
uv run python -m calib_auto.golden import --source PATH [--name NAME] [--format auto|png|jpeg|keep]
uv run python -m calib_auto.golden stats NAME

uv run python -m calib_auto.glyphs                       # build glyph caches
uv run python -m calib_auto.classifier build             # train the template bank
uv run python -m calib_auto.classifier eval --split photo
uv run python -m calib_auto.classifier eval --split set --holdout run-011
uv run python -m calib_auto.classifier read PHOTO        # probe one photo
uv run python -m calib_auto.train_cnn                    # CNN: validation + artifact

uv run python -m calib_auto.segment PHOTO --json         # detector debug
```

## Tests

```sh
uv run python -m pytest
```

The calibration suite runs the full loop against in-memory fakes
(`tests/fakes.py`: a physical display model with firmware-accurate
clamps and caps, a simulated reader); the CNN/benchmark suites use
synthetic segmentable photos (`tests/synthutil.py`). No camera,
display, network or GPU needed.

## Provenance

Ported from the three earlier tools (`tools/calib`, `tools/calib-vlm`,
`tools/ocr-vlm`) — this package shares no runtime code with them and is
self-contained. The calibration loop is the calib-vlm loop with the
recognizer generalized; the glyph/classifier/CNN stack is the ocr-vlm
classifier work; the display client, camera and VLM client are the
calib/calib-vlm implementations.
