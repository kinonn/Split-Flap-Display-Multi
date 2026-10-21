"""VLM-reader auto-calibration for the modular split-flap display.

Unlike tools/calib (seam heuristics + cross-module consensus), the
feedback signal here is a vision-language model *reading* the display:
one photo in, a fixed per-module schema out (glyph + condition), then
deterministic offset math on top of the known drum order.
"""
