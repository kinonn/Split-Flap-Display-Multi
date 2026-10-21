"""VLM read benchmark for split-flap display photos.

Point the tool at a directory holding a ``reads.jsonl`` dataset (one
``{photo, want, saw}`` record per line, photos beside the file), let a
vision-language model transcribe each photo through calib-vlm's proven
per-module reader, and score the transcription position by position
against the commanded frame. The recorded ``saw`` value is scored the
same way as the baseline, so every run answers one question: is this
provider/model reading better or worse than the reader that produced the
dataset?
"""
