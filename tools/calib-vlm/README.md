# splitflap-calib-vlm

VLM-reader auto-calibration for the modular split-flap display.

Unlike `tools/calib` (seam heuristics + cross-module consensus, which
produced wrong verdicts), the feedback signal here is a vision-language
model *reading* the display. One photo in, a fixed per-module schema out
(glyph + flap condition), then deterministic offset math on top:

- Drums move forward-only and `GET /api/calib/status` gives `drumOrder`,
  so "module shows X but you commanded Y" converts directly into
  `(drumIndex(Y) - drumIndex(X)) * stepsPerChar` forward steps.
- The display has exactly `numModules`/`totalModules` modules, so the
  reader is required to report one entry per position and leading/
  trailing blanks are reconciled against the known width (never trimmed).
  Photos can be annotated with a module grid + index ticks to help.

The VLM only reads. All tuning, budgets, previews/persists, snapshot
rollback and acceptance are deterministic Python.

## Setup (one time)

```sh
cd tools/calib-vlm
uv sync
```

## Run

```sh
uv run calib-vlm-server   # http://127.0.0.1:8002
```

In the browser:

1. **Configure** — display host (master), camera index, reader provider
   preset (OpenCode Go / OpenAI / OpenRouter / Ollama / custom), model,
   API key (stored server-side, mode 0600, never shown back), camera
   brightness/crop/exposure/start-wait, module-grid annotation. Start-wait (0..30 s, default 5) is the warm-up patience
   for cameras whose driver opens showing all-black frames for a
   while — raise it if the check fails with "produced only black
   frames".
2. **Read test** — type a pattern, show it, and see the reader's exact
   per-module transcription with expected-vs-read chips.
3. **Run** — pick a mode plus the phases to run (P1, P2, P4,
   acceptance checkboxes; P0 registration always runs first):
   - `dry-run`: P0 + one reverse uniform sweep, then the per-module
     shift/purity table (no writes).
   - `full`: the selected phases in fixed order (coarse module offsets ->
     fine char cells -> verify/acceptance), committing each phase after
     it verifies. P2 without P1 first runs a read-only sweep to find
     suspect characters (no commits). Skipping acceptance can never
     report `converged` — subset runs end `needs-human` for review.
     Phase selection applies to `full` mode; `dry-run` always sweeps only.
4. **Report** — `converged` or `needs-human` with reasons, per-module
   deltas/readings and `report.json` in the run dir.

State (config, photos, reports, snapshots) lives in `./data`
(`CALIB_VLM_DATA` overrides it); `CALIB_VLM_HOST`/`CALIB_VLM_PORT`
override the listen address.

## How the reader works

Each frame is one VLM call returning `report_reading` with exactly one
entry per module, left to right:

| field | meaning |
| --- | --- |
| `char` | the glyph that module actually shows (space = blank flap) |
| `condition` | `clean`, `half`, `double`, `blank`, `unreadable` |
| `confidence` | 0..1 for the character reading |

The prompt is blind (the commanded frame is never shown to the model), so
verdicts are independent. It states the exact module count and forbids
trimming leading/trailing blanks. If the model still returns a compacted
answer (e.g. `AB` for `  AB  `), `reader.py` aligns the visible run to the
known width against the expected blank layout, fills the rest, and flags
every position as `inferred` + `realigned` (never silently trusted). An
answer *longer* than the display gets one corrective re-ask. On repeated
reader failures (3 in a row) the run stops; a single failure degrades to
an all-`unreadable` reading so the phase can escalate instead of crash.

## Calibration steps

The run is a deterministic state machine (`calib_vlm/calibrate.py`):
P0 -> reverse uniform sweep (P1 coarse) -> P2 fine -> P4 verify -> acceptance.
Fixes are applied as volatile previews, verified, then committed **per phase**
(module offsets once P1 confirms them, char cells once P2 confirms). A full
`/settings` snapshot is saved first (manual rollback via the UI), and hold is
engaged at the start and released in a `finally`.

### P0 — registration

1. Show a blank frame, then all-`H`, then an index strip
   (`ABCDEFGHIJ.../0123456789`).
2. Read each; compacted blanks on the blank/all-`H` frames are fine
   (reconciled), but the index strip is strict: a collapsed answer, an
   all-unreadable strip, or a read order that looks mirrored fails the
   run with `P0 registration failed`.

Purpose: prove the camera maps left-to-right to module 0..N-1 and that
the display is readable before anything is tuned.

### P1 — coarse: reverse uniform sweep -> module offsets

One uniform frame per character (`c * total`), walking the drum **backwards**
(`%`, `#`, `@`, ... `A`, `space`). Each step is ~a full revolution, so every
frame passes the magnet and re-homes: every sample is independent. Because all
modules are commanded the same character, a module that disagrees is either
misaligned or misread.

Per module, build the histogram of the signed shift
`s = drumIndex(seen) - drumIndex(commanded)` (trusted reads only; confusable
pairs excluded):

- **mode >= 80% of >= 24 trusted samples** -> one-shot whole-character module
  fix `delta = mode * stepsPerChar`.
- **below the purity gate but >= 50% of trusted samples on a single ±1
  residual** -> systematic fault, fixed on the module cell the same way
  (escalating it would dead-end every affected character in P2: the
  firmware's ±32 char-cell clamp can never hold a whole-character fix).
- **single-flap arc** (dominant state correct, but >= 12.5% of the drum
  shows the same ±1 residual) -> the module enters the sub-pitch trim
  ladder with an extra proportionate candidate
  `round(arc_count / drumLen * stepsPerChar)`; the ladder keeps it only if
  it beats the baseline without breaking a guard, otherwise the arc is
  flagged for P2. Blank reads against non-blank commands are junk samples
  and excluded from the histogram entirely.
- **worse than that** -> the reads are unreliable: re-read the deviant frames
  once, then escalate `needs-human` (reader/hardware), never "correct" noise.
- **a few same-sign +/-1 residuals** (a minority of the module's characters)
  -> a **sub-pitch module trim**: candidates `P/2, P/4, P/8, P/16`
  (`P = stepsPerChar`), cumulative in the outlier direction, scored on the
  outlier characters plus spread guard characters; the smallest shift that
  clears every outlier without breaking a guard wins.
- **several `half`/`double` cells** -> a module-cell phase search
  (`+/-1/2/4/8`) centres the seam.

Files still wrong or unreadable after P1 become P2 work. Whole-character fixes
and trim winners are committed as module cells before P2 starts.

### P2 — fine: per-character offsets

Re-reads the flagged characters (identity mismatches and seam cells) at the
committed P1 base and walks **every** char-cell fault — a wrong glyph
included — up the **incremental ladder** (parallel across cells): candidate
offsets `+/-4/8/12/...` motor steps in increments of 4 up to the firmware's
`+/-32` char-cell clamp, then the narrow-window fallback `+/-2, +/-1`. Each
candidate is applied from the base and kept only when it beats the cell's
baseline without breaking a guard, so the smallest offset that reads
correct+clean wins; a cell that reads clean stops being probed (the rest of
its scan could only tie it). A one-shot exact identity delta is never
written: a fault that needs a whole flap is escalated as hardware, reporting
the probes it tried, rather than sent as a clamped, wrong offset.

Char-cell deltas are chunked to +/-32 within one batch pass, the flagged frames
are re-read to verify, and the confirmed char cells are committed.

### P4 — verify: repeatability + folded border check

Repeats a few uniform frames and requires identical reads, then walks a sample
of **short forward hops** across drum boundaries (every 7th character), checking
character and condition — the reverse sweep only exercises near-full
revolutions, so binding/double-flap on small moves needs this pass.

### Acceptance

A forward uniform sweep over the whole drum. Pass only when every module reads
every non-space character with a clean flap, `space` reads blank, no escalation
remains, and the repeats were stable. Confusable-only differences never fail
(the glyphs are indistinguishable on the drum).

`converged` is only possible in `full` mode; `dry-run` only sweeps and reports
the shift/purity table (no writes).

### Offset math and tuning primitives

- `stepsPerChar = round(stepsPerRot / len(drum))`, read from the
  `/settings` snapshot. Identity fixes apply the **signed-minimal** drum
  distance (`(index(target) - index(seen))` wrapped to ±half the drum) in
  motor steps: one char ahead is −1 char, not a near-full revolution
  forward. An offset correction re-anchors the firmware's character
  table (then re-homes) — it is not a drum move, so the drum's
  forward-only rule does not apply.
- Module-cell corrections are **negated**. A per-char offset shifts that
  character forward for positive steps, but a module offset re-anchors
  the homing magnet reference (`position = magnetPos + moduleOffset` on
  magnet detection, then forward-only steps to `charPosition`), so a
  positive module offset shows an *earlier* drum character. Getting this
  backwards makes the tune walk the drum away from the target.
- Firmware limits are honoured exactly: char-cell previews are chunked to
  `|delta| <= 32` and char cells are clamped to ±32, so any char-cell fix
  that would exceed that range is escalated rather than written clamped.
  Module (`charIndex = -1`) offsets are unbounded (up to a full
  revolution per call), so a whole-character module correction is sent as
  **one** preview that re-homes the module once instead of once per 32
  steps. A `preview-batch` call carries at most 48 nudges (the endpoint
  rejects more with HTTP 400) and at most 8 nudges per remote group (the
  master's loop-task drain forwards only the first 8, dropping the rest
  silently), so each scope is chunked to those caps.
- Cells are tuned by a parallel **cell ladder**: every plan applies its own
  candidate in the same mixed frames/reads (one `preview-batch` per step),
  and the score is `target read correct+clean` per target minus a penalty
  per broken guard glyph, so a candidate is kept only when it beats the
  cell's baseline — a non-improving nudge is reverted before the next
  step. A cell that reaches a perfect score stops being probed (later
  candidates could only tie it) and an exhausted cell holds at its best
  offset. Candidates are coarse-to-fine: sub-pitch fractions of one
  character for boundary flaps, `+4, -4, +2, -2, +8, -8, +1, -1` motor
  steps for a whole-drum seam (identity right, several cells reading
  half/double), the incremental `+/-4/8/12/16/20` scan (increments of 4 up
  to half a character pitch, `+/-2, +/-1` as the narrow-window fallback)
  for a per-character `half`/`double` flap, and the exact signed-minimal
  identity delta for a per-character P2 fault.
- P1 runs a **parallel sub-pitch module trim** before any per-character
  cell is touched: a module wrong on only a few glyphs is usually a
  boundary/phase problem, so candidates `0.75/0.5/0.25/0.125` of one
  character are tried on the module cell, scored by the target glyphs
  plus spread "guard" glyphs (so a trim that fixes the boundary by
  breaking the neighbours is rejected). Suspect modules are independent,
  so each applies its own candidate in the same mixed frames/reads and
  `POST /api/calib/preview-batch` applies them all and re-homes the
  touched modules in **one** pass — cost is `rounds x frames`, not
  `modules x rounds x frames`. Batches with `scope: 2..6` are forwarded by
  the master over ESP-NOW and applied RAM-only on the remote group (no NVS
  write; the group acks when homing finishes), and a winning remote trim is
  persisted once via `/api/calib/offsets`.
- Local group (1): volatile `/api/calib/preview` nudges, re-read, then
  persist the verified absolute value via `/api/calib/offsets`.
- Remote groups: volatile fleet preview (above) for the trim; other cell
  work uses persist-verify-revert. The initial base comes from the master's
  `rModOffs` / `rChrOff0..4` settings (the `status` endpoint only exposes
  local live offsets), but every commit persists the **tracked absolute**
  (base + all preview deltas applied to that cell), never the run-start
  snapshot: the persisted value must equal the value the device holds, or
  a second commit on the same cell (P1's module trim and phase trim
  overlap) would drop the first verified component.
- Baseline hygiene: every run starts with `POST /api/calib/reload`, which
  reverts any RAM-only preview residue from earlier runs; `full` mode does
  the same on the way out (persisted winners remain, ghost residue on
  escalated cells is dropped). Without this, live offsets from an
  aborted/preview run are read as the baseline and the tuner chases
  corrections that do not exist (the settings rollback is a no-op because
  it only reloads on an actual change).

### Fleet, modes and budgets

- All shows go to the master; fleet-width frames fan out via ESP-NOW.
  Engage hold on remote controllers out-of-band before starting.
- Fleet geometry: the per-group module counts come from the status
  endpoint's `groupWidths` list (group 1 first), else the
  `masterGroupModuleCounts` CSV of the `/settings` snapshot — the same
  vector the firmware maps groups with, so an asymmetric fleet (e.g.
  `8,6,4`) is addressed correctly. Firmware that reports neither falls
  back to equal-width groups with a short last group.
- `dry-run` sweeps and prints the shift/purity table only (no writes);
  `full` applies, verifies and commits each phase.
- Budgets: frames 300/1200, reader calls 300/1200, previews 200/800,
  persists 400/1600, sweeps 3/8, plus a 90-minute (`max_seconds`,
  default 5400) wall-clock cap and abort checks throughout. Exceeding a
  budget stops the run; already-committed phases are kept (commit per
  phase), volatile residue is reverted, and `snapshot.json` remains for
  a manual rollback. A cell ladder that cannot afford another probe
  round stops early and commits the offsets it has already verified, so
  a nearly-spent preview/frame budget costs the untested candidates and
  not the winners found so far.

## Run artifacts

Each run writes to `data/runs/run-NNN/`:

- `events.jsonl` — the full event log (read/tune/phase/error).
- `<tag>_f<frameId>.png` — every photo (raw, not annotated).
- `snapshot.json` — the pre-run `/settings` body for rollback.
- `report.json` — result/reason, fleet geometry, per-frame readings and
  warnings, every delta (old/new, before/after, proposal/fixed),
  persistent escalations, per-module summary, budget usage and
  `timing.phases` (one row per phase boundary — seconds + frames/VLM/
  preview/persist deltas, each attributed to the phase that produced it).

## Code map

- `calib_vlm/vlm.py` — OpenAI-compatible vision chat client (no SDK).
- `calib_vlm/prompts.py` — reader system prompt + `report_reading` tool
  schema.
- `calib_vlm/reader.py` — parsing + reconciliation to the known module
  count.
- `calib_vlm/calibrate.py` — the sweep/fine/verify state machine, offset math,
  fleet semantics and budgets.
- `calib_vlm/server.py` — FastAPI app, masked API key (0600), run
  harness, read-test endpoint.
- Reuses `splitflap-calib` for `Display`, `Camera` and the contract.

## Tests

```sh
uv run pytest
```
