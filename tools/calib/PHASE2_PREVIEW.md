# Phase 2 — volatile preview (dry run, no persist)

Requires Phase 1. Preview nudges live RAM offsets on ONE local module and
re-homes only that module. Nothing is written to NVS. `POST
/api/calib/reload {}` (or a settings save that actually changes a
calibration value, or a reboot) reverts previews. Note a settings POST that
re-sends identical values does NOT reload — use the reload endpoint to
guarantee a clean baseline.

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
- `delta`: non-zero motor steps. For char cells (`charIndex >= 0`) it is
  `-32..32` and positive moves the flap forward along the drum. For the
  coarse module offset (`charIndex = -1`) it may be up to a full
  revolution, and the sign is inverted on the drum (a positive module
  offset shifts the displayed character *backward*): apply a whole
  correction in one call so the module only re-homes once.
- `409` = display busy; back off and poll.

Batch form (preferred when nudging several independent modules):
`POST /api/calib/preview-batch {"nudges":[{scope?:1..6, module, charIndex, delta},...]}`
applies every nudge and re-homes all touched modules in one pass. Scope 1
is the local controller; scopes 2..6 are forwarded by the master over
ESP-NOW and applied RAM-only on that remote group (it acks when homing
finishes, so the master's `busy` covers it). The same `/api/calib/reload`
(or offsets push) reverts remote previews too.

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
- To revert everything: `POST /api/calib/reload {}` discards all previews
  (reboot also works, and so does a settings save that actually changes a
  calibration value). Call reload at the start of a session so residue
  from a previous aborted run is never read as the baseline. State this in
  your report.
