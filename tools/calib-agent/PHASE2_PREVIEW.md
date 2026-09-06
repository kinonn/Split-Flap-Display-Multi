# Phase 2 — volatile preview (dry run, no persist)

Requires Phase 1. Preview nudges live RAM offsets on ONE local module and
re-homes only that module. Nothing is written to NVS. Any reload
(`POST /settings` offset save, reboot, or new persist) reverts previews.

## Protocol

```
POST /api/calib/preview {"module": 2, "charIndex": -1, "delta": 2}
-> 202 {module, charIndex, delta}
... poll status until busy==false, re-show the test glyph, photo ...
```

- `module`: 0-based local index on the controller you call. For fleets,
  call each group's controller IP directly (preview is local-only).
- `charIndex`: `-1` = coarse module offset, else drum index
  `0..charset-1` into `drumOrder` (NOT ASCII). Bounds-check against
  `charset` from status.
- `delta`: non-zero motor steps, `-32..32`. Positive moves the flap
  forward along the drum.
- `409` = display busy; back off and poll.

## Loop

1. Show the suspect glyph (Phase 1 sync), photo = before.
2. Preview one delta on one module only.
3. Re-show the same glyph, photo = after. Keep the delta only if the
   flap visibly improves and neighbors do not regress.
4. One variable at a time; log
   `{module, charIndex, delta, beforePhoto, afterPhoto, kept}`.

## Rules

- Coarse (`charIndex:-1`) before per-character.
- Max +/-32 per nudge; re-home settles before photographing.
- Never preview on two modules concurrently; never pipeline previews
  while `busy==true`.
- To revert everything: reboot is NOT needed — any persisted offset save
  reloads NVS and discards previews. State this in your report.
