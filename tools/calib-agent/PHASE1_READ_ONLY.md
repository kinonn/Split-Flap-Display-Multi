# Phase 1 — read-only vision (no offset writes)

You can move the display and photograph it. You MUST NOT write offsets,
change settings, reboot, or touch MQTT/ESP-NOW config in this phase.

## 0. Preconditions

- Fixed camera, whole display in one frame, locked exposure/white balance.
- Agent on LAN; controllers reachable by HTTP (master + each group IP).
- Read `contract.json` (or `GET /calib-contract.json`): drum order, limits,
  pattern list. Never hardcode glyph order.

## 1. Snapshot

```
GET /api/calib/status
```

Record: `numModules`, `totalModules`, `groupCount`, `charset`,
`drumOrder`, `mode`, `holdActive`, `busy`, `frameId`. Abort if
`contractVersion`/`schemaVersion`/`charset` mismatch your bundle.

## 2. Hold

```
POST /api/calib/hold {"active": true}
```

On EVERY controller (master + each remote IP). Verify `holdActive==true`.
Hold (mode 4) suspends date/time/random/scroll/MQTT/ESP-NOW writes.

## 3. Sync protocol (every photo)

```
POST /api/calib/show {"frame": "<exact width>", "dwellMs": 800}
-> 202 {frameId, fleetFrame}
GET /api/calib/status ... poll until busy==false
... wait dwellMs ...
... shoot photo ...
GET /api/calib/frame?frameId=N  (ground truth, settled==true to use)
```

- Frame length MUST equal `numModules` (local show) or `totalModules`
  (fleet-wide show via master, distributed left-to-right). `409` = busy,
  `400` = wrong width.
- Drums move forward-only: issue patterns in drum order.

## 4. Pattern sequence

- **P0 registration:** blank (all spaces) -> all `H` (exposure/ROI) ->
  index strip `ABCDEFGH...` / `01234567` (camera X -> module map, detect
  mirroring, wrong order, dead modules; spans all groups).
- **P1 coarse (6 moves, all modules same):** `[" ","E","H","O","0","-"]`.
  Glyphs with top/bottom bars expose vertical half-flap best. Estimate
  global `displayOffset` first, then per-module coarse error.
- **P2 fine strided:** staggered sweep — module `i` shows
  `drumOrder[(k+i) % N]`, step `k` in drum order. One photo covers N
  distinct flaps across the fleet. Full rotation = 37/48 stops, not
  37 x modules.
- **P3 boundaries:** adjacent drum neighbors (`A/B`, `M/N`) for
  double-flap / binding.
- **P4 repeatability:** home + same char 3x for lost steps / hall drift.

## 5. What to check per photo

- ROI stable, per-module crop from P0 map.
- Half-flap (gap at top/bottom), double-flap (two glyphs visible),
  stuck module (no change across frames), wrong module order.
- Log: `{frameId, frame, photoFile, perModule:[{module, seen, ok, note}]}`.

## 6. Finish

```
POST /api/calib/hold {"active": false}
```

On every controller. Report JSON; propose offsets in text only — do not
call `/settings`, `/api/calib/preview`, or `/api/calib/offsets`.
