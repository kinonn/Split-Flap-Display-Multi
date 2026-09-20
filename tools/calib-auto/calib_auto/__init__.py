"""calib-auto: unified standalone auto-calibration tool.

Two calibration approaches behind one loop (the reader is pluggable):

- **CNN** — a small model trained on your golden set reads every frame
  locally (no provider, no network, fast).
- **VLM** — a vision-language model reads frames through the
  ``report_reading`` tool call (provider-configured, slower, stronger
  on frames the CNN has never seen).

Supporting tools, all in this package:

- **Golden set** — curate labeled photos (the ground truth for both
  training and testing); tracked by git.
- **Train** — build glyph caches from the golden set and train/retrain
  the CNN (or the dependency-free template bank) from the web UI.
- **Benchmark** — run any configured VLM against the golden set to
  decide which model is most suitable for the VLM approach.
"""

__version__ = "0.1.0"
