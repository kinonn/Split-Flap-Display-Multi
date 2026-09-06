# PRODUCTION — vision-guided auto-calibration runbook (single instruction set)

Fixed camera framing the whole display. Agent on LAN over HTTP.
Fleet from day one: master + ESP-NOW remotes, up to 6 groups x 8 modules.
Firmware: `feature/ai-calibration-apis`, contract v1, settings schema v1.

Abort unless `GET /api/calib/status` and `GET /calib-contract.json`
match this pin (contractVersion 1, charset and module counts as expected).

## 1. Safety envelope

- Drums move forward-only: issue all patterns in `drumOrder` sequence
  (from status), never random.
- Shows are exact-width: `len == numModules` (local) or `== totalModules`
  (fleet-wide via master, fanned left-to-right). Centering/scroll forced off.
- Hold (mode 4) suspends date/time/random/scroll/MQTT/ESP-NOW writes:
  `POST /api/calib/hold {"active":true}` on EVERY controller first,
  `{"active":false}` on all when done.
- Wear budget: max 3 full rotations per module per session.
- Pre-flight: snapshot via `GET /settings` (store the full response body).
  before any persist. Rollback = `POST /settings` with the snapshot's inner `settings` object.

## 2. Sync protocol (every photo)

```
POST /api/calib/show {"frame": "<exact width>", "dwellMs": 800} -> 202 {frameId}
GET /api/calib/status -> poll until busy==false; wait dwellMs; shoot
GET /api/calib/frame?frameId=N -> ground truth (settled==true to use)
```

## 3. Pattern sequence

- **P0 register:** blank -> all `H` (exposure/ROI) -> index strip
  `ABCDEFGH...`/`01234567` (camera-X to module map, mirror/order/dead check).
- **P1 coarse (all modules same):** `[" ","E","H","O","0","-"]` ->
  fit `displayOffset` globally, then `moduleOffsets` per module.
- **P2 fine (staggered sweep):** module `i` = `drumOrder[(k+i) % N]`,
  step `k` in drum order (stride 6, then refine suspects).
- **P3 boundaries:** adjacent drum neighbors (`A/B`, `M/N`) for binding.
- **P4 repeatability:** home + same char 3x, pixel-stable per crop.

## 4. Decide -> preview -> persist

```
POST /api/calib/preview {"module":m, "charIndex":-1|0..N-1, "delta":d}
  # volatile, RAM-only, local controller only (call each group IP); reload reverts
POST /api/calib/offsets {"scope":"local"|2..6, "kind":"char"|"module"|"display", ...}
  # one cell, NVS + selective re-home / ESP-NOW push; re-photo after each
```

Order: display -> module -> char; local group first, then remotes.
`charIndex -1` = coarse module offset, else drum index into `drumOrder`.
Char values clamped `-32..32`.

## 5. Done criteria + report

Converged when edge glyphs are clean on all modules, staggered sweep
matches, 3x repeats stable. Emit
`{contractVersion, frames:[{frameId, frame, photo}], deltas:[...],
beforeAfter, result}`. On anomaly (`busy` stuck, dead module,
regression): stop, unhold all, report `needs-human`.
