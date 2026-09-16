# splitflap-ocr-vlm

Benchmark harness for VLM reads of split-flap display photos.

Point it at a directory holding a `reads.jsonl` dataset (one record per
line, photos beside the file), let a vision-language model transcribe
every photo (per-module tool call or plain OCR prompt — see *Read
modes*), and score the transcription position by position against the
commanded frame. The recorded `saw` value is scored the same way as the
baseline, so one run answers: **is this provider/model better than the
reader that produced the dataset?**

## Dataset format

```jsonl
{"photo": "sw_37_f255.png", "want": "%%%%%%%%%%%%", "saw": "%%%%%%%#%%%%"}
```

- `photo` — image file in the same directory as the JSONL file
- `want` — the frame that was commanded when the photo was taken
- `saw` — what the baseline reader reported (scored as the comparison)

Rows are always 12 characters for the known fleets (one per module, a
blank module is a space; the width is per row and only flagged, not
enforced).

## Quick start

```sh
cd tools/ocr-vlm
uv sync
uv run ocr-vlm-server          # → http://127.0.0.1:8003
```

Then in the browser:

1. **Configure** — provider preset (OpenCode Go / OpenAI / OpenRouter /
   Ollama / custom), model, API key, charset (48-char drum set by
   default) and the number of parallel VLM calls.
2. **Dataset** — type the directory (or use *Discover run dirs*, which
   scans `../calib-vlm/data/runs/*` and `../calib-agent/app/data/runs/*`
   for `reads.jsonl` files) and hit *Validate*.
3. **Run** — optionally cap the run to the first N rows (quick test),
   then *Start benchmark*.
4. **Summary / Rows / Report** — compare VLM vs baseline totals, walk the
   per-row table, inspect individual reads (photo + per-module chips),
   download report JSON / rows CSV.

## What gets compared

Score of a row = number of character positions where the read differs
from `want` (lower is better). Rows are normalized first (uppercase,
`␣`/`_`/`space`/… → space, pure decoration removed — a lone `'` is a
drum character and survives). A value that needed padding or truncation
to the row width is flagged `adjusted` and still counted. Rows that
failed (invalid line, missing photo, VLM error) are excluded from every
average and counted separately.

The report includes, per method (VLM and `saw`): total / average /
median mismatches, exact-match count and rate, a mismatch histogram, and
per-position mismatch rates; plus head-to-head row counts (better /
equal / worse), top confusion pairs, and a breakdown by photo-name
prefix (`sw_`, `p0_`, `ladder_`, …).

The VLM read is **blind**: `want` is never shown to the model, and the
reader's reconciliation gets no expected frame (`expected=""`), so the
ground truth cannot leak into the alignment either.

## Read modes

Not every vision model can make function calls — an OCR specialist such
as PaddleOCR-VL ignores the `report_reading` tool entirely and the tool
reader fails every frame with *"model did not call report_reading"*.
The `Read mode` selector picks how photos are read:

| Mode | Path | Use for |
| --- | --- | --- |
| `auto` (default) | Probe with the tool reader on the first read of a run; on failure switch that reader to text mode for good | any model |
| `tool` | calib-vlm's per-module `report_reading` tool call | VLMs with function calling |
| `text` | Native OCR prompt (default `OCR:`) + text parse | OCR models (PaddleOCR-VL, …) |

In text mode the reply is parsed as characters: whitespace-separated
glyphs (`% % % % # %`), contiguous runs (`ABCDEFGHIJKL`) and merged
chunks (`MMMJN% % BBCN`) all work, `space`/`blank` words count as one
blank module, and a reply shorter/longer than the row is padded /
truncated and flagged `adjusted` (rows are never dropped). The raw
reply is kept with each row and shown in the inspect card, and rows read
via an `auto` fallback carry a `text-fallback` flag. An empty reply
counts as an all-blank read (`no-text` flag) — for a blank frame that is
correct, for a missed frame it is a real 12-mismatch miss, not a hidden
error. The OCR prompt itself is editable in the UI, because prompt
style is the critical knob for OCR models (PaddleOCR-VL answers
instruction-style prompts with hallucinated document structure but
transcribes cleanly for its native `OCR:` token).

Image preparation differs per path, deliberately: the tool reader gets
the same annotated photo as the production reads (module separators +
index ticks drawn on top), the text reader gets the untouched photo — a
probed PaddleOCR-VL server answers an annotated photo by transcribing
the index digits (`0 1 2 ... 11`) instead of the display underneath.
The text path also has its own image settings (max width, JPEG quality,
PNG/JPEG format) in the Configure card: the dataset photos are
1280x260 natively, so raising the width above the 1024 default and/or
using PNG sends the OCR model more of the original detail. Measured on
the run-011 dataset (DeepSeek-OCR-2-8bit, 155 rows): 1280 px / q95 JPEG
scored best (avg 3.03 mismatches vs 3.20 at 1024 px / q80); lossless PNG
was no better than high-quality JPEG.

Generation is capped per request (`Max output tokens`, default 128,
0 = uncapped). Greedy decoding produces the same first tokens with or
without a cap, so this never changes a score (the parser only reads the
first characters) — but a model that degenerates into a repetition loop
(observed on rows of one repeated character, re-emitting the glyph until
the server's 2048-token limit) drops from ~13 s to ~1.6 s per row.

Note the comparison is slightly conservative for the VLM: the `saw`
baseline was produced with the commanded frame available for
reconciliation, while every read here is blind.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `OCR_VLM_DATA` | `./data` | config + run artifacts |
| `OCR_VLM_HOST` | `127.0.0.1` | bind host |
| `OCR_VLM_PORT` | `8003` | bind port |
| `OCR_VLM_DATASET_DIR` | — | overrides configured dataset dir |
| `OCR_VLM_DATASET_FILE` | — | overrides configured dataset file |
| `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` | — | provider overrides |

The API key is stored server-side only (`data/config.json`, mode 0600)
and is never rendered back — the UI shows `***last4`.

## Run artifacts

Each run gets `data/runs/run-NNN/` with:

- `events.jsonl` — live log (same events the UI streams)
- `results.json` — every row incl. per-module detail
- `report.json` — status, config, dataset fingerprint (sha256), summary

## Code map

| File | Role |
| --- | --- |
| `ocr_vlm/dataset.py` | JSONL parsing/validation, photo guard, discovery |
| `ocr_vlm/score.py` | normalization, positional scoring, aggregates |
| `ocr_vlm/textread.py` | OCR text read path + tool/text/auto dispatch |
| `ocr_vlm/server.py` | FastAPI routes, config, run harness (threads) |
| `ocr_vlm/static/index.html` | the whole UI (vanilla JS, no build step) |

The tool reader itself is imported from `splitflap-calib-vlm`
(`VlmReader` / `VLMClient` / `prompts`) so this tool and the
calibration loop never drift apart; the text path reuses the same
charset normalization for identical semantics.

## Tests

```sh
uv run pytest
```

Covers normalization/scoring edge cases (apostrophes, wrappers, blank
words), the OCR text parser and the tool/text auto dispatch (including
the per-path image preparation), dataset validation markers, the
photo-name guard, config masking, and a full run against a fake reader
(no network needed).
