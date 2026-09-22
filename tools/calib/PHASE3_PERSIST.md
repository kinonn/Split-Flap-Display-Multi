# Phase 3 — scoped persist (fleet push)

Requires Phases 1-2. Writes ONE offset cell to NVS and applies it
(selective re-home locally, ESP-NOW push for remotes).

## Pre-flight (mandatory)

1. Snapshot via API only (no browser): `GET /settings` and store the
   full response body. This is your rollback.
2. Confirm hold is engaged on all controllers.

## Protocol

```
POST /api/calib/offsets
{"scope": "local", "kind": "char", "module": 2, "charIndex": 5, "value": 3}
{"scope": "local", "kind": "module", "module": 2, "value": -4}
{"scope": "local", "kind": "display", "value": 2}
{"scope": 2, "kind": "char", "module": 1, "charIndex": 5, "value": 3}
-> 200 {message, scope, kind}
```

- `scope`: `"local"` (or `1`) = controller you call; `2..6` = remote
  group on the MASTER (master fans out via ESP-NOW push).
- `kind`: `char` (needs `module`, `charIndex`, `value -32..32`),
  `module` (needs `module`, `value`), `display` (needs `value`).
- One cell per call. After each call, re-show the affected glyph fleet-
  wide and photograph before the next write.
- Order: global `display` -> coarse `module` -> per-character `char`;
  local group first, then remotes 2..N.

## Rollback

Re-post the snapshot's inner `settings` object via `POST /settings`
(or re-`POST` the previous cell values to `/api/calib/offsets`). Verify
with a P1 coarse re-show.

## Rules

- Never use full `POST /settings` for calibration (it risks WiFi/MQTT/
  hardware rewrites) — use this scoped endpoint.
- Commit only photo-verified values from Phase 2.
