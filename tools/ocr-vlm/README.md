# splitflap-ocr-vlm

Benchmark harness for VLM reads of split-flap display photos.

Point it at a directory holding a `reads.jsonl` dataset (one record per
line, photos beside the file), let a vision-language model transcribe
every photo (per-module tool call or plain OCR prompt — see *Read
modes*), and score the transcription position by position against the
commanded frame. The recorded `saw` value is scored the same way as the
baseline, so one run answers: **is this provider/model better than the
reader that produced the dataset?**

`reads.jsonl` records what the display was *commanded* to show — which
is not necessarily what a photo shows when the display itself needs
calibration. For assessing OCR accuracy, curate a **baseline set**
instead: the same photos with their content verified by eye on the
`/curate` page. Baseline sets score models against the *actual visible
content* and keep the previous reader's answer as the comparison
baseline — and they are also the training data for the local glyph
classifier (`read_mode=classify`), which reads photos with no provider
at all (see *Local glyph classifier*).

## Dataset format

```jsonl
{"photo": "sw_37_f255.png", "want": "%%%%%%%%%%%%", "saw": "%%%%%%%#%%%%"}
```

- `photo` — image file in the same directory as the JSONL file
- `want` — the frame that was commanded when the photo was taken
- `saw` — what the baseline reader reported (scored as the comparison)

Rows are always 12 characters for the known fleets (one per module, a
blank module is a space; the width is per row and only flagged, not
enforced). The benchmark UI sources its datasets from curated
**baseline sets** (see below); `reads.jsonl` files remain supported
through the API and the tests.

## Baseline sets (curation)

A baseline set is ground truth for OCR accuracy: copies of the photos
plus the content actually visible in them, entered by a human. Create
one on the **Curate** page (`/curate`, linked from the benchmark) from
a calibration run dir (its `report.json` frames) or any `reads.jsonl`
dir, then work through the entries:

- `content` is pre-filled with whatever the previous run recorded
  (`read` from a calib-vlm report, `saw` from a `reads.jsonl`) — a
  typing aid that may well be wrong; correct it.
- Saving an entry marks it **verified**. Entries start **pending** and
  are excluded from runs by default, so a half-curated set can never
  silently pass pre-fills off as truth. *Mark pending* reverts an
  entry; *Next pending* walks the unchecked ones; the header shows
  `verified X / Y`.
- The editor spells spaces visually (`␣`) because blank flaps are real
  positions, shows the commanded `want` and prior `prior_read` for
  reference only, and flags every position you changed vs the pre-fill.

Layout under `tools/ocr-vlm/baselines/<name>/`:

```
set.json         # name, created, source run, module_count
baseline.jsonl   # {"photo","content","status","prior_read","want","tag","frame_id"}
images/          # byte-identical copies of the source photos
```

Photos are copied, so a set survives cleanup of its source run and is
self-contained (the first set, imported from a 122-frame run, is about
45 MB — commit it if the ground truth is worth sharing).
`OCR_VLM_BASELINES` overrides the root folder.

To benchmark against a set: pick it in the benchmark's *Baseline set*
selector — the only dataset source the UI offers. Sets with no verified
entry are listed but not selectable until curated, and every run uses
verified content only (pending entries are skipped and counted as
`skipped_pending` in Validate and the run report). Scoring uses
`content` as truth and `prior_read` as the baseline column. *Discard*
on the curate page deletes a whole set; *Remove image* deletes a single
entry. (`reads.jsonl` datasets remain supported by the API and the
tests, but the UI no longer exposes them.)

## Quick start

```sh
cd tools/ocr-vlm
uv sync
uv run ocr-vlm-server          # → http://127.0.0.1:8003
```

**Local classifier in three commands** — the fastest path to
provider-free reads (requires curated baseline sets; see *Local glyph
classifier* for the full walkthrough and troubleshooting):

```sh
uv run python -m ocr_vlm.glyphs                                          # 1. labeled glyph caches from verified entries
uv run python -m ocr_vlm.train_cnn                                       # 2. validation metrics, then data/glyphs/cnn.pt
uv run python -m ocr_vlm.classifier eval --split photo                   # 3. honest quality check before trusting it
```

Then in the UI: *Read mode* = **Local classifier…**, pick a baseline
set, *Start benchmark* — no provider fields, no API key.

For VLM benchmarking instead, in the browser:

1. **Configure** — provider preset (OpenCode Go / OpenAI / OpenRouter /
   Ollama / custom), model, API key, charset (48-char drum set by
   default), the number of parallel VLM calls, and — for the OCR text
   path — read mode, segmentation, preprocess style and debug images
   (see *Read modes* below). For the `classify` read mode no provider
   is needed at all: train the local model first (see *Local glyph
   classifier*), then pick it in the Read mode selector.
2. **Baseline dataset** — pick a validated baseline set from the
   selector (create or curate sets on the `/curate` page); the
   selection is validated automatically. Sets with no verified entry
   cannot be selected.
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
| `classify` | Local trained model, one classification per module cell — no provider, no network | the display's own font (see *Local glyph classifier*) |

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

Photo segmentation (`Segmentation` in the Configure card) picks how the
photo reaches the model. `strip` sends the whole photo (default).
`cells` crops each module on the known 12-column grid and sends one
glyph per request — every reply maps to an exact module position and
run/merge errors are impossible — at the cost of one request per module.
Cells mode is model-dependent: it was tested against DeepSeek-OCR-2 and
*hurt* badly (9.4 vs 2.8 errs on the same 5 photos) because an isolated
flap crop is out of distribution for a document OCR model — blank cells
draw hallucinated output (a blank cell once came back as "The 7 Habits
of Highly Effective People", another as a CJK character). Its grounding
prompt (`<|grounding|>OCR this image.`) fixes the format on single
crops (replies become `<|ref|>%<|/ref|><|det|>[[box]]<|/det|>`) but
blanks still hallucinate. Treat cells mode as an experiment to A/B
against a given model; the Past runs table compares the results.

### Detected modes (OpenCV segmentation)

`cells` splits the *whole frame* into equal columns, which in the
recorded datasets is wrong: the display occupies roughly x=160..1205 of
a 1280 px photo (module pitch ≈87 px, not 107), so the legacy split puts
every crop off-centre and edge cells mostly on background. The detected
modes locate the display first (`ocr_vlm/segment.py`, classical CV only
— no ML, no network) and use the real module grid:

| Mode | Path | Use for |
| --- | --- | --- |
| `montage` | 12 normalized crops in one labeled contact sheet, one request | models that read a grid better than a thin strip |
| `cells-detect` | one non-blank module crop per request (blank cells are never sent) | A/B against `cells` |
| `strip-detect` | the display cropped out of the frame (optionally normalized), one request | A/B against `strip` |

How it detects: the middle band's per-column median gray is bimodal
(display vs wall/fixture), so an Otsu split yields the display's columns;
within them a row counts as display when a large fraction is near-black
(the bright glyph band is bridged by a gap-tolerant run). On run-011 the
detector found the display on **155/155** photos at ±1 px width
variation. When detection fails the read falls back to the plain strip
path and the row is flagged `no-detect` — a bad detector degrades to the
old numbers, never to garbage crops.

`Preprocess` (detected modes) styles every crop before sending it:
`none` (as captured), `contrast` (CLAHE), `invert` and `binary` (Otsu
black-on-white, document-like — the models are trained on dark text on
light paper). Blank flaps are decided locally by ink fraction, so blank
cells never reach the model and cannot draw hallucinated output; in
`montage`, a blank cell the model claims to have read is overridden by
the local decision (flagged in the row warnings). All of this is a
setting, not an assumption: A/B it per model in the Past runs table.

Two debug aids: the Dataset card's **preview** renders the current
segmentation/`Preprocess` settings for any photo in the configured
dataset without running a model, and **Debug images** saves the composed
image of every row under `runs/run-NNN/composed/`, shown in the inspect
card (with a live preview fallback when debug images are off).

Note the comparison is slightly conservative for the VLM: the `saw`
baseline was produced with the commanded frame available for
reconciliation, while every read here is blind.

## Local glyph classifier (`read_mode=classify`)

Every VLM tested so far misreads this display's font in the same
font-specific ways — `0` as `O`, `1` as `7`, `/` as `I` — because the
glyphs are out of distribution for models trained on documents, not
because the font is ambiguous (the zero is dotted, the one is flagged;
photos show it clearly). The fix is a recognizer trained on the font
itself, and the curated baseline sets are its training data: every
verified entry pairs a photo with the human-checked content, so each
module crop is a labeled glyph sample.

How a read works: the display is located with the same OpenCV detector
the other detected modes use, each cell is blank-gated, and non-blank
cells are classified — one character per module, with per-cell
confidence and margin. No provider, no API key, no network; a full run
of 142 photos takes under a minute on CPU.

### The pipeline

1. **Glyph caches** (`glyphs.py`) — walk every **verified** entry of
every baseline set, crop each module, and reduce the crop to the
glyph's own bounding box (`segment.canonical_glyph`: bright ink,
specular lip glints filtered out, padded and squared). This geometry
normalization is what makes glyphs from different rigs, exposures and
band heights comparable — raw module windows are not (the same `7`
scored 0.60 against its own per-class mean before it). Caches land in
`data/glyphs/<set>.npz` with a JSON summary (per-class counts, skipped
photos and why).
2. **A model** — two interchangeable backends behind one
`predict_raster` interface:
   - **bank** (`classifier.py`, no extra deps): per-class mean of
     unit-normalized rasters, cosine match. The dependency-free
     fallback and the A/B baseline for the CNN.
   - **cnn** (`train_cnn.py`, PyTorch CPU): a small conv net on the
     same rasters with OpenCV/numpy augmentation (tilt, scale,
     brightness, blur, noise, erase — mirroring the captures' real
     defects: motion-ghosted flaps, mid-transition tilts, glints).
     Validated photo-disjointly, then retrained on all curated cells
     as the deployment artifact (`data/glyphs/cnn.pt`).
3. **The reader** (`classify_reader.py`) — display detection, blank
gating, per-cell classification, and honest flags: blank cells are
decided locally (no model call), cells under the confidence/margin
floor are counted and named in the row warnings (`low-conf:N` flag),
and a photo with no detected display is an all-blank read flagged
`no-detect` (classify mode has no strip fallback — never silent
garbage). Cells the ink test calls blank but that carry bright compact
ink are still classified, which is what rescues the period and
apostrophe (their ink area is under the blank threshold).

### How to use it

**Step 1 — curate the training data.** The classifier learns only from
**verified** baseline-set entries; pending pre-fills are never used. If
you have not curated yet, import a calibration run on the `/curate`
page and type the visible content for each photo (see *Baseline sets*
above). A few hundred verified photos (≈3,000 module cells) already
train a strong model — more data only helps, especially for the
glyph pairs your runs still confuse.

**Step 2 — build the glyph caches** (rerun after any curation change):

```sh
uv run python -m ocr_vlm.glyphs
```

```
glyph caches in .../tools/ocr-vlm/data/glyphs
  run-005: 108 cells from 9 photos
  run-011: 1704 cells from 142 photos
  run-001: 1248 cells from 104 photos
total: 3060 cells, 255 photos, 0 skipped
```

One `<set>.npz` plus a summary JSON per set lands in `data/glyphs/`.
Photos that cannot contribute (missing file, no detected display,
content width ≠ module count) are skipped with a reason and counted,
so a data problem is visible instead of silently shrinking the set.

**Step 3 — train a model.** The CNN is the recommended backend and
needs no GPU (a few minutes on CPU):

```sh
uv run python -m ocr_vlm.train_cnn
```

It runs twice, by design: first a **photo-disjoint validation run**
(20% of the photos are held out by a hash of their names, so the
reported accuracy is never on a photo the model trained on; it prints
per-epoch loss and validation accuracy), then a **final run on all
curated cells** that becomes the deployment artifact
`data/glyphs/cnn.pt` (+ `cnn.json` with the validation metrics and
provenance). The dependency-free template bank trains in seconds and
remains available as the fallback and as an A/B baseline:

```sh
uv run python -m ocr_vlm.classifier build      # → data/glyphs/bank.npz
```

**Step 4 — check the quality numbers** (recommended before trusting
the model):

```sh
uv run python -m ocr_vlm.classifier eval --split photo                  # photo-disjoint
uv run python -m ocr_vlm.classifier eval --split set --holdout run-011  # leave-one-shoot-out
uv run python -m ocr_vlm.classifier read path/to/photo.png              # probe one photo
```

Example output (leave-one-shoot-out, bank backend):

```
split=set holdout=run-011 · size=48 · bank classes=48
  train: 1356 cells from run-001, run-005
  test : 1704 cells / 142 photos from run-011
  per-cell: bank 0.940 · reader-sim 0.955
  rows: 96/142 exact (0.676) · avg mismatches 0.5352
  low-confidence cells (<0.9): 339 (0.199)
  confusions: ' '->'-' 22, ':'->' ' 8, 'M'->'.' 6
```

How to read it: `reader-sim` is the production number (blank gating +
classification); `avg mismatches` is the unit the benchmark scores in.
The `--split set` run is the honest "new shoot" estimate — it trains on
the other sets and tests on one the model has never seen — and the
confusion list names exactly which glyph pairs to curate more of (or
that the CNN still cannot separate). `--split photo` is stricter but
smaller; `--split all` is an in-sample sanity check only.

**Step 5 — use it in the benchmark.** In the UI: *Configure* → **Read
mode = Local classifier…** (provider fields are ignored; no API key
needed) → *Baseline dataset* → pick a validated set → *Start
benchmark*. Rows then read `classify`; the inspect card shows each
module's character and confidence; the Summary card's pipeline line
names the exact model artifact:

```
read=classify · seg=display+canonical64 · prep=none · model=cnn:9eeb686b sets=run-001,run-005,run-011 · conf>=0.5 margin>=0.1 · modules=12
```

Via the API instead of the UI:

```sh
curl -X POST http://127.0.0.1:8003/api/config -H "Content-Type: application/json" \
  -d '{"dataset_dir": ".../baselines/run-011", "dataset_file": "baseline.jsonl", "read_mode": "classify"}'
curl -X POST http://127.0.0.1:8003/api/run/start -H "Content-Type: application/json" -d '{}'
```

**Step 6 — retrain as data grows.** After curating more photos, rerun
steps 2–3. The artifacts are replaced and every future run's pipeline
record carries the new model hash, so old and new runs stay comparable
in the Past-runs table.

### Configuration

`read_mode=classify` needs no provider fields. In the config (API; the
UI exposes the read mode):

| Key | Default | Meaning |
| --- | --- | --- |
| `classifier_backend` | `auto` | `auto` prefers the CNN artifact when present, else the bank; or force `bank` / `cnn` |
| `classifier_model` | — | explicit artifact path override (`.pt` = CNN, otherwise bank npz) |
| `classifier_min_conf` | `0.5` | flag classified cells below this softmax/cosine score |
| `classifier_min_margin` | `0.1` | flag classified cells whose top1−top2 gap is below this |

### Measured results

- Leave-one-shoot-out (train on run-001 + run-005, test on the
  never-seen run-011, bank backend): **95.4% per-cell, 0.54 mismatches
  per row** — close to the production reader's 0.458 on its own
  training data.
- CNN, photo-disjoint split: 93.9% per-cell, 84.6% row-exact.
- Deployment run (`run-055`, CNN, curated run-011 set, 142 photos):
  **1 mismatch total (avg 0.007) vs the production reader's 65 (avg
  0.458)** — 124 rows equal, 17 better, 1 worse; 0 errors, 1
  low-confidence row, 0.33 s per photo.

This mode is the local ground-truth reader the benchmark needed: it
scores near-perfectly on curated data and its remaining errors are
concentrated on genuinely hard captures (mid-transition flaps), which
the per-cell confidence flags surface instead of hiding.

### Troubleshooting

| Symptom | Meaning / fix |
| --- | --- |
| `no glyph caches found` | Run step 2; also check the sets actually have verified entries (pending-only sets are reported, not trained) |
| `local classifier not usable: no CNN model at …` | No artifact yet — run step 3, or train the bank and set `classifier_backend: "bank"` |
| `excluded labels (not in charset): 'v' x1` | A curated typo (found in run-001); fix that entry on `/curate`, then rebuild (step 2) |
| Rows flagged `low-conf:N` | N cells scored under `classifier_min_conf` / `classifier_min_margin` — inspect them; usually mid-transition captures or capture conditions the training data lacks |
| Rows flagged `no-detect` | The display was not located; classify mode has no strip fallback, so the row is an honest all-blank read, not silent garbage |
| Wrong backend used | `classifier_backend` defaults to `auto` (CNN preferred when `cnn.pt` exists); force `bank`/`cnn`, or point `classifier_model` at an artifact path (`.pt` = CNN) |
| Everything reads as blanks | The selected set's photos may lack a detectable display — run `uv run python -m ocr_vlm.segment PHOTO --json` to inspect the detection |

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `OCR_VLM_DATA` | `./data` | config + run artifacts |
| `OCR_VLM_BASELINES` | `./baselines` | root folder of curated baseline sets |
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
- `report.json` — status, config, dataset fingerprint (sha256), summary,
  and a `pipeline` block (`text` + `parts`) recording how a photo became
  characters: read mode, segmentation, preprocess, the text path's image
  encoding + prompt + token cap, and the tool path's fixed production
  encoding (annotated photo, 1024 px q80 JPEG) — enough to recreate the
  exact pipeline in code. Classify runs add a `classifier` part: backend
  (`bank`/`cnn`), artifact path + **sha256**, the sets it was trained on,
  and the confidence/margin floors
- `composed/` — the composed image of every row, only when `Debug
  images` is on (each row's `composed` field names its file)

## Code map

| File | Role |
| --- | --- |
| `ocr_vlm/dataset.py` | JSONL parsing/validation, baseline-set loading, photo guard, discovery |
| `ocr_vlm/curate.py` | baseline-set curation: create/read/update/remove/discard, photo copies |
| `ocr_vlm/score.py` | normalization, positional scoring, aggregates |
| `ocr_vlm/segment.py` | OpenCV display detection, module grid, blank gating, canonical glyph rasters, normalization, montage (+ CLI) |
| `ocr_vlm/textread.py` | OCR text read path, per-glyph cell segmentation, detected modes, tool/text/auto dispatch |
| `ocr_vlm/glyphs.py` | glyph-cache builder: verified baseline sets → labeled canonical rasters (+ CLI) |
| `ocr_vlm/classifier.py` | template-bank backend, blank/glyph gating policy, evaluation splits, confusion report (+ CLI) |
| `ocr_vlm/train_cnn.py` | CNN backend: augmentation, training, photo-disjoint metrics, artifact save/load (+ CLI) |
| `ocr_vlm/classify_reader.py` | `read_mode=classify` reader: local model, per-cell confidence, low-conf flags |
| `ocr_vlm/server.py` | FastAPI routes, config, run harness (threads) |
| `ocr_vlm/static/index.html` | the whole benchmark UI (vanilla JS, no build step) |
| `ocr_vlm/static/curate.html` | the curation UI (create sets, verify content, discard) |

The tool reader itself is imported from `splitflap-calib-vlm`
(`VlmReader` / `VLMClient` / `prompts`) so this tool and the
calibration loop never drift apart; the text path reuses the same
charset normalization for identical semantics.

`segment.py` imports only cv2 + numpy, deliberately: it is
extraction-ready for a shared library (the natural promotion target is
`calib_vlm`) or a standalone tool. It overlaps with
`tools/calib/calib/vision.py`'s `split_crops` (equal strips nudged to
seams); unify on promotion instead of growing a third dialect.

Standalone validation without a model or server:

```sh
uv run python -m ocr_vlm.segment ../calib-vlm/data/runs/run-011          # dataset stats
uv run python -m ocr_vlm.segment ../calib-vlm/data/runs/run-011/shot.png --json
```

## Tests

```sh
uv run pytest
```

Covers normalization/scoring edge cases (apostrophes, wrappers, blank
words), the OCR text parser and the tool/text auto dispatch (including
the per-path image preparation), dataset validation markers, the
photo-name guard, config masking, segmentation (synthetic split-flap
frames: detection, module boxes, seam snapping, blank gating, montage
layout, CLI), the detected read modes (montage blank override,
cells-detect blank skipping, stripless fallback, preprocess styling),
the preview endpoint and debug artifacts, glyph-cache extraction
(verified-only, skips with reasons, cache roundtrip), the classifier
(bank fit/predict, charset exclusion, blank-gating policy, evaluation
splits), the CNN backend (tiny-data training, artifact roundtrip) and
the classify reader (end-to-end reads, low-confidence flags, missing
artifacts), plus full runs against a fake reader and a real local
classifier (no network needed).
