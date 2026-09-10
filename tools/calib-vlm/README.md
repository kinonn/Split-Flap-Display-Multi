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
   API key (stored server-side, mode 0600, never shown back), reasoning
   effort, camera brightness/crop/exposure, module-grid annotation.
2. **Read test** — type a pattern, show it, and see the reader's exact
   per-module transcription with expected-vs-read chips.
3. **Run** — pick a mode and a sweep:
   - Mode: `dry-run` (read + propose), `preview` (volatile nudges),
     `full` (persist offsets, converge when verified).
   - Sweep: **Fast / sampled** (default, two staggered passes) or
     **Exhaustive per-character** (every drum character on every module;
     slower, more motor wear).
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
P0 -> P1 -> P2 -> P3 -> P4 -> acceptance, all under preview/persist
budgets. Before any write a full `/settings` snapshot is saved; hold is
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

### P1 — coarse, module level first

Shows one uniform frame per test glyph. The set starts from the
easy-to-read `E H O 0 -` and expands with an even spread across the drum
until it covers at least **25% of the character set** (10 glyphs on the
37-char drum, 12 on the 48-char drum; blanks are skipped because a blank
flap carries no identity information).

Every glyph is read on every module and mismatches are counted per module
(trusted reads only). Faults are routed by vote:

- **>= 2 wrong glyphs** = a whole-drum fault -> tune the **module
  offset** (`charIndex = -1`, coarse).
- **exactly 1 wrong glyph** = a per-character fault -> tune that
  character's **char cell** (`charIndex = drum index of the glyph`).
  Tuning a running drum coarse for a single-glyph fault would break the
  module's other characters. If the needed correction does not fit the
  firmware's ±32-step char-offset range (a whole-character fix is
  `stepsPerChar` motor steps, 55 at the default settings), the module
  escalates `needs-human` instead — a clamped, wrong cell value would
  leave the display worse than before.

A verify pass re-reads all five glyphs and applies the same split; the
final all-`H` check escalates any module still wrong or unreadable.

### P2 — fine, per character

Staggered sweep in drum order: `frame[i] = drumOrder[(k + i) % N]` for
`k` advancing in steps of 6.

- **Fast / sampled (default):** two passes (`k` at offsets 0 and 3).
- **Exhaustive per-character:** all six residue classes, so every drum
  character lands on every module at least once.

For each frame the reader returns a distinct glyph per module, which
gives three detections:

- **Wrong glyph** -> per-char identity tune for that character.
- **`half` / `double`** -> per-char alignment tune (small-step search).
- **Stuck**: a module reads the same trusted glyph across consecutive
  frames that command *different* glyphs (majority of comparisons) ->
  escalate `needs-human` (a frozen module can never converge).

### P3 — drum-neighbour boundaries

Shows frames alternating adjacent drum neighbours (e.g. `A/B`, `M/N` in
7-step jumps) to catch binding / double-flap at character boundaries:
wrong glyph -> per-char identity tune, `double` -> alignment tune.

### P4 — repeatability

Shows all-`H` three times. Any module whose character or condition
differs between repeats is escalated as unstable (lost steps / hall
drift).

### Acceptance

1. No persistent escalation may remain.
2. Show `E`, then `H`; every module must read the commanded glyph with
   condition `clean` and confidence at or above the configured minimum.

`converged` is only possible in `full` mode; `dry-run`/`preview` always
end `needs-human` (preview may pass acceptance but nothing was persisted).

### Offset math and tuning primitives

- `stepsPerChar = round(stepsPerRot / len(drum))`, read from the
  `/settings` snapshot. Identity fixes apply
  `drum_delta(seen, target) * stepsPerChar` forward motor steps, so a
  one-character error is corrected in one move (not by the old ±1..8
  step search).
- Firmware limits are honoured exactly: the `/api/calib/preview` handler
  rejects a single `|delta| > 32` motor steps, so larger corrections are
  applied as a sequence of bounded preview calls; char cells are clamped
  to ±32, so any char-cell fix that would exceed that range is escalated
  rather than written clamped. Module (`charIndex = -1`) cells are
  unconstrained and absorb whole-drum corrections.
- Alignment fixes search `+4, -4, +2, -2, +8, -8, +1, -1` motor steps,
  each candidate applied from the base (not compounded).
- Cost per reading: wrong character, then condition rank
  (`clean`/`blank` < `half` < `double` < `stuck` < `unreadable`), then
  confidence.
- Local group (1): volatile `/api/calib/preview` nudges, re-read, then
  persist the verified absolute value via `/api/calib/offsets`.
- Remote groups: no preview in firmware, so persist-verify-revert per
  cell. Absolute bases come from the master's `rModOffs` / `rChrOff0..4`
  settings (the `status` endpoint only exposes local live offsets).

### Fleet, modes and budgets

- All shows go to the master; fleet-width frames fan out via ESP-NOW.
  Engage hold on remote controllers out-of-band before starting.
- `dry-run` reads and records proposals only (no preview, no persist);
  `preview` applies local volatile nudges; `full` persists.
- Budgets (fast / exhaustive): frames 300/1200, reader calls 300/1200,
  previews 200/800, persists 400/1600, sweeps 3/8, plus a 1-hour
  wall-clock cap and abort checks throughout. Exceeding a budget raises
  and rolls the display back to the pre-run snapshot.

## Run artifacts

Each run writes to `data/runs/run-NNN/`:

- `events.jsonl` — the full event log (read/tune/phase/error).
- `<tag>_f<frameId>.png` — every photo (raw, not annotated).
- `snapshot.json` — the pre-run `/settings` body for rollback.
- `report.json` — result/reason, fleet geometry, per-frame readings and
  warnings, every delta (old/new, before/after, proposal/fixed),
  persistent escalations, per-module summary and budget usage.

## Code map

- `calib_vlm/vlm.py` — OpenAI-compatible vision chat client (no SDK).
- `calib_vlm/prompts.py` — reader system prompt + `report_reading` tool
  schema.
- `calib_vlm/reader.py` — parsing + reconciliation to the known module
  count.
- `calib_vlm/calibrate.py` — the P0..P4 state machine, offset math,
  fleet semantics and budgets.
- `calib_vlm/server.py` — FastAPI app, masked API key (0600), run
  harness, read-test endpoint.
- Reuses `splitflap-calib` for `Display`, `Camera` and the contract.

## Tests

```sh
uv run pytest
```
