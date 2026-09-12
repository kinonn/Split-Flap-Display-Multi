# VLM calibration harness web app

Containerized ReAct agent: a vision-capable LLM drives the display's
calibration APIs (`/api/calib/*`) with a USB camera watching, inside
structural guardrails (P0-first, preview-before-persist, budgets,
classical acceptance gate). You configure only: display host, LLM
provider/base URL, model, API key.

## Setup and run with uv

```sh
cd tools/calib-agent/app
uv sync
```

Configure once — either via environment:

```sh
export DISPLAY_HOST=splitflap.local
export LLM_BASE_URL=https://opencode.ai/zen/go/v1   # default: OpenCode Go (key from OpenCode Zen)
export LLM_MODEL=deepseek-v4-flash-vision-exp   # or OpenRouter, Ollama, vLLM…
export LLM_API_KEY=sk-...
uv run calib-agent-server   # http://127.0.0.1:8000
```

…or open the web UI and fill the Configure card (key is stored
server-side, mode 0600, never shown back). Then: Check display, Check
camera, pick a mode, Start.

Modes (Run card selector, enforced by the harness, mirrored from the
deterministic CLI phases): **dry-run** (show + photograph only,
proposals in text — safe first run), **preview** (+volatile offset
nudges, nothing saved), **full** (+persist offsets, converge when the
acceptance gate passes).

State (config, photos, reports, template bank) lives in `./data` by
default (`CALIB_AGENT_DATA` overrides it).

## How it works

- `agent_app/vlm.py` — OpenAI-compatible vision chat client (function
  calling, base64-JPEG image parts). One code path for OpenAI,
  OpenRouter, Ollama, vLLM.
- `agent_app/agent.py` — ReAct loop + tools (`get_status, hold, show,
  capture, preview, persist, finish`) + guardrails the model cannot
  override. Every tool result bundles classical vision scores with the
  photo so the VLM fuses both. System prompt is assembled from
  `../PRODUCTION.md` + `../PROMPT_SNIPPET.md` (single source of truth).
- `agent_app/server.py` — FastAPI UI + config API (key masked, 0600 file),
  run control, live log/photos/report, snapshot restore.
- Safety: step/persist budgets, P0-before-tuning, local persist needs a
  preview+photo first, `finish(converged)` validated by the classical
  acceptance gate, hold always released at end, snapshot saved per run.

## Tests

```sh
cd tools/calib-agent/app
uv run pytest
```
