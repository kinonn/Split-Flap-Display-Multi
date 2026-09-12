# AI Agent Calibration — index

Vision-guided auto-calibration for the modular split-flap display.
Fixed camera framing the whole display, agent on LAN calling HTTP APIs.
Fleet from day one: 1 master + ESP-NOW remotes, up to 6 groups x 8 modules.

## Which files to give the agent

| Stage | Hand the agent |
| --- | --- |
| Production (all phases done) | `PRODUCTION.md` + `contract.json` (pinned same version) |
| Phase 1 only | `PHASE1_READ_ONLY.md` + `contract.json` |
| Phase 1+2 | add `PHASE2_PREVIEW.md` |
| Phase 1-3 | add `PHASE3_PERSIST.md` |
| All | add `PHASE4_AUTONOMY.md` (`PRODUCTION.md` supersedes phases) |

`PROMPT_SNIPPET.md` is a copy-paste system prompt. Never hand the agent
the full `src/` tree, WiFi/MQTT secrets, or NVS dumps.

## Version matrix

| Instruction bundle | Firmware | contract | settings schema |
| --- | --- | --- | --- |
| v1 (this drop) | `feature/ai-calibration-apis` | 1 | 1 |

Live truth always wins: `GET /api/calib/status` and
`GET /calib-contract.json` on the device must match the header pin in
`PRODUCTION.md` before moving anything.

## Automated runners (no chat needed)

Two programs implement this spec. Prefer them over manual copy-paste
sessions.

### VLM harness web app (recommended) — `app/`

A vision-capable LLM drives the calibration APIs with a USB camera
watching, inside structural guardrails (P0-first, preview-before-persist,
budgets, classical acceptance gate). You configure only four things:
display host, LLM provider, model, API key. Defaults: OpenCode Go
(`https://opencode.ai/zen/go/v1`, key from OpenCode Zen) with
`deepseek-v4-flash-vision-exp`.

```sh
cd tools/calib-agent/app
uv sync
export DISPLAY_HOST=splitflap.local
export LLM_BASE_URL=https://opencode.ai/zen/go/v1
export LLM_MODEL=deepseek-v4-flash-vision-exp
export LLM_API_KEY=<your-key>
uv run calib-agent-server   # → http://127.0.0.1:8000
```

Then in the browser:

1. **Configure** — fill display host, provider preset (or custom
   OpenAI-compatible URL), model, API key (stored server-side, mode
   0600, never shown back), camera index. Save.
2. **Check display** — verifies the master is reachable and reports
   module count / charset / contract version. Abort on mismatch.
3. **Check camera** — verifies a stable, well-exposed frame (fails fast
   on drift, saturation, or a display out of frame).
4. **Start** — the agent engages hold on the display and works P0→P4.
   Watch the live log and photo gallery; use **Abort** any time, and
   **Restore pre-run snapshot** to roll back NVS to the pre-run state.
5. **Report** — `converged` (acceptance: clean edge glyphs, identity
   verified, repeats stable) or `needs-human` with reasons, evidence
   photos, and per-cell deltas. Everything also lands on disk under the
   run directory (`agent_report.json` + photos + `snapshot.json`).

Safety notes: step/persist budgets are enforced by the harness, local
persists require a prior preview+photo cycle, `finish(converged)` is
validated by the classical acceptance gate (not by the model), and hold
is always released at run end. Still, supervise the first run against
real hardware before leaving it unattended. Any OpenAI-compatible
provider works (OpenAI, OpenRouter, Ollama, vLLM) — the model just needs
vision input plus function calling.

### Deterministic CLI (no LLM) — `tools/calib/`

Same spec as a fixed state machine, no model involved:

```sh
cd tools/calib
uv sync
uv run splitflap-calib --host splitflap.local --photo-dir ./photos          # full run
uv run splitflap-calib --host splitflap.local --phase 1 --photo-dir ./p1    # dry run
```

See `tools/calib/README.md` for flags (`--camera-index`, `--phase`,
`--preview-only`, `--check-camera`).
