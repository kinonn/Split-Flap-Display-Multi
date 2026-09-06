# Phase 4 — full autonomy loop

Requires Phases 1-3. The agent runs the whole calibration without human
approval, within the budgets below.

## State machine

```
snapshot = GET /settings (store body)                      # rollback point
hold(all controllers)
P0 register -> module map (abort on dead/stuck module: report, do not tune around silently)
P1 coarse: show [" ","E","H","O","0","-"] on all modules
  -> fit displayOffset globally, then moduleOffsets per module (persist via Phase 3, local first)
P2 fine: staggered drum-order sweep, strided (every 6th) then refine suspects
  -> preview (Phase 2) each candidate, persist winners (Phase 3)
P3 boundaries: neighbor pairs, fix double-flap stragglers
P4 repeatability: home + same char 3x
P1 re-show fleet-wide = acceptance
unhold(all), publish report
```

## Convergence (all must hold)

- Edge glyphs (`E`, `H`, `O`, `0`, `-`) clean on every module, no
  half-flap gap, no double-flap.
- Staggered sweep shows the expected glyph per module per frame.
- 3x repeat of the same char is pixel-stable per module crop.

## Budgets / safety

- Max 3 full drum rotations per module per session (wear).
- One outstanding show/preview at a time; `409 busy` = back off 2s.
- Any I2C/hall anomaly (module never settles, `busy` stuck > 60s):
  stop, unhold, report `needs-human` with logs — do not keep pushing
  offsets.
- Rollback on regression: `POST /settings` with the snapshot's inner
  `settings` object when acceptance is worse
  than the pre-flight P1 photo set.

## Report (JSON)

```
{contractVersion, schemaVersion, charset, fleet:{groups, modulesPerGroup},
 frames:[{frameId, frame, photo}], deltas:[{scope, kind, module, charIndex, old, new}],
 beforeAfter:{p1Before:[photos], p1After:[photos]}, result:"converged|needs-human"}
```
