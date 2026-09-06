# splitflap-calib

Vision-guided auto-calibration runner for the modular split-flap display.
You provide the master hostname/IP; a USB camera on this host watches the
display. No browser needed — everything goes through the display's HTTP
APIs (`/api/calib/*`, `/settings`). Fleet-ready: the master fans show
frames and offset pushes out to ESP-NOW remotes.

Spec: `../calib-agent/PRODUCTION.md` (+ `PHASE1..4`).

## Setup (one time)

```sh
cd tools/calib
uv sync
```

## Run

```sh
# 1. Camera + device preflight (no display writes beyond test frames)
uv run splitflap-calib --host splitflap.local --check-camera

# 2. Safe dry run: Phase 1 read-only, proposal report only
uv run splitflap-calib --host splitflap.local --phase 1 --photo-dir ./photos

# 3. Full auto-calibration (preview -> persist -> acceptance)
uv run splitflap-calib --host splitflap.local --photo-dir ./photos
```

Useful flags: `--camera-index 0`, `--phase {1,2,3,4}` (default `4`),
`--dwell-ms 800`, `--contract PATH` (override bundled
`calib/contract.json`), `--timeout-s 60`.

## How it works

- `calib/display.py` — HTTP client for the master (status/hold/show/frame/
  preview/offsets + settings snapshot/restore for rollback).
- `calib/camera.py` — USB capture, exposure lock attempt, stability check.
- `calib/vision.py` — no ML: P0 index strip gives per-module crops; each
  crop is scored for half-flap seam (full-width horizontal edge in the
  middle band), double-flap (two full-width seams), and stuck modules
  (no change between frames). Glyph *identity* is verified by
  cross-module consensus: on uniform frames every clean crop must look
  alike (zero-mean normalized cross-correlation), so a clean-but-wrong
  glyph is flagged, re-verified with a dedicated show, and escalated to
  `needs-human` (offsets cannot fix a whole-glyph error). Tune with
  `--identity-thresh`. An absolute template bank (`photo-dir/templates/`,
  one reference crop per glyph) additionally checks each crop against the
  *commanded* glyph, catching systematic shifts where all modules agree
  yet are all wrong. First run bootstraps it from consensus winners;
  golden banks are never auto-extended — refresh explicitly with
  `--relearn-templates` after a verified calibration.
- `calib/loop.py` — P0 register -> P1 coarse -> P2 fine -> P3 boundaries ->
  P4 repeatability -> acceptance, with wear/time budgets. Writes
  `report.json` into `--photo-dir`.
- Group 1 uses preview-then-persist; remote groups (master-only access)
  use persist-verify-revert per cell.

## Tests

```sh
uv run pytest
```
