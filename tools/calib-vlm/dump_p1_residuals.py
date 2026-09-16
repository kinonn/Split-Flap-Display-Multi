#!/usr/bin/env python3
"""Dump the P1 residual map of a calib-vlm run.

`calibrate.py` computes the P1 residual map (per module, per character:
the signed flap shift left after the reverse uniform sweep and the deviant
re-read pass) as a local variable and never persists it, so the only record
of *why* P2 picked its flagged characters lives in the run's photos and the
`sw_*` / `p1r_*` frames. This script replays the exact P1 algorithm against
`report.json` and prints that map.

It is forensic-only: it reads `report.json`, touches nothing, and works on
any run directory.

Usage:
    uv run python dump_p1_residuals.py                 # latest run
    uv run python dump_p1_residuals.py run-006
    uv run python dump_p1_residuals.py data/runs/run-006 --json
    uv run python dump_p1_residuals.py --list

Run it with plain `python` too: the calibrate constants are imported when
the calib-vlm environment is available and embedded as a fallback
otherwise (the script itself is stdlib-only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# Constants mirrored from calib_vlm/calibrate.py. Imported when the package
# (and its cv2 / splitflap-calib deps) is installed; embedded otherwise so the
# script stays runnable under a bare interpreter. Keep in sync on change.
_FALLBACK = {
    "confusables": {
        "O": ("0", "D", "Q"), "0": ("O", "D", "Q"),
        "D": ("O", "0"), "Q": ("O", "0"),
        "I": ("1",), "1": ("I",),
        "S": ("5",), "5": ("S",),
        "Z": ("2",), "2": ("Z",),
        "B": ("8",), "8": ("B",),
        "G": ("6",), "6": ("G",),
    },
    "min_samples": 24,        # SWEEP_MIN_SAMPLES
    "trust_purity": 0.80,     # SWEEP_TRUST_PURITY
    "majority_share": 0.50,   # SWEEP_MAJORITY_SHARE
    "arc_share": 0.125,       # SWEEP_ARC_SHARE
    "skip_chars": (".", "'", "-"),  # DEFAULT_SKIP_CHARS
}


def _load_spec() -> dict:
    try:
        from calib_vlm import calibrate as C
    except Exception:
        return dict(_FALLBACK)
    return {
        "confusables": C.CONFUSABLES,
        "min_samples": C.SWEEP_MIN_SAMPLES,
        "trust_purity": C.SWEEP_TRUST_PURITY,
        "majority_share": C.SWEEP_MAJORITY_SHARE,
        "arc_share": C.SWEEP_ARC_SHARE,
        "skip_chars": C.DEFAULT_SKIP_CHARS,
    }


def _confusable(a: str, b: str, spec: dict) -> bool:
    return a != b and b in spec["confusables"].get(a, ())


def _trusted(entry: dict | None, min_conf: float) -> bool:
    """calibrate.VlmCalibrator._trusted on a report.json module entry."""
    return bool(
        entry
        and entry.get("source") == "vlm"
        and entry.get("char") not in (None, "")
        # '?' is the unknown sentinel but also a real drum character.
        and (entry["char"] != "?" or entry.get("expected") == "?")
        and entry.get("condition") != "unreadable"
        and entry.get("confidence", 0) >= min_conf
    )


def _drum_delta(drum: str, seen: str, target: str) -> int | None:
    """calibrate.VlmCalibrator._drum_delta: signed-minimal char distance."""
    if seen not in drum or target not in drum:
        return None
    n = len(drum)
    raw = (drum.index(target) - drum.index(seen)) % n
    return raw - n if raw > n // 2 else raw


def _shift(entry: dict | None, ch: str, drum: str, min_conf: float,
           spec: dict) -> int | None:
    """calibrate.VlmCalibrator._shift: signed shift (characters)."""
    if (entry is None or not _trusted(entry, min_conf)
            or entry["char"] not in drum or ch not in drum
            or _confusable(entry["char"], ch, spec)):
        return None
    if entry["char"] == " " and ch != " ":
        return None
    delta = _drum_delta(drum, entry["char"], ch)
    return None if delta is None else -delta


def _mode_purity(values) -> tuple[int | None, float, int]:
    counts = Counter(v for v in values if v is not None)
    if not counts:
        return None, 0.0, 0
    mode, count = counts.most_common(1)[0]
    return mode, count / sum(counts.values()), sum(counts.values())


def _modes_for(drum: str, shifts: dict, spec: dict) -> dict[int, int]:
    """calibrate.VlmCalibrator._p1_coarse module classification.

    Returns {module: mode}; a module missing from the map is one the run
    escalated as unreliable (it contributes nothing to the residual map).
    """
    modes: dict[int, int] = {}
    for m in sorted(shifts):
        mode, purity, total = _mode_purity(list(shifts[m].values()))
        if total < spec["min_samples"] or purity < spec["trust_purity"]:
            # Systematic majority fault fixable on the module cell.
            if (total >= spec["min_samples"] and mode is not None
                    and abs(mode) == 1 and purity >= spec["majority_share"]):
                modes[m] = mode
                continue
            # Single-flap arc: dominant state correct, same-sign ±1 arc.
            arc = [(ch, s) for ch, s in shifts[m].items()
                   if s is not None and abs(s) == 1]
            if (mode == 0 and total >= spec["min_samples"]
                    and len(arc) / max(1, len(drum)) >= spec["arc_share"]
                    and len({s < 0 for _, s in arc}) == 1):
                modes[m] = 0
                continue
            continue  # escalated -> no residual entry
        modes[m] = mode
    return modes


def _frames_by_tag(report: dict) -> dict[str, dict]:
    return {f.get("tag"): f for f in report.get("frames", [])}


def _readings_from(frame: dict) -> dict[int, dict]:
    return {md["module"]: md for md in frame.get("modules", [])}


def reconstruct(report: dict, spec: dict | None = None) -> dict:
    """Rebuild the P1 residual map from a report.json body."""
    spec = spec or _load_spec()
    fleet = report.get("fleet", {})
    drum = str(fleet.get("drumOrder", ""))
    total = int(fleet.get("totalModules", 0))
    cfg = report.get("config", {})
    min_conf = float(cfg.get("min_confidence", 0.0))
    skip = set(cfg.get("skip_chars", "")) if cfg.get("skip_enabled", True) \
        else set()

    frames = _frames_by_tag(report)
    if not drum:
        raise SystemExit("report has no fleet.drumOrder; not a tuning run")

    # Initial reverse uniform sweep.
    readings: dict[int, dict] = {}
    chars: list[str] = []
    for tag, frame in frames.items():
        if not tag.startswith("sw_"):
            continue
        frame_txt = frame.get("frame") or ""
        if not frame_txt:
            continue
        ch = frame_txt[0]
        if ch not in drum or ch in skip:
            continue
        chars.append(ch)
        for m, md in _readings_from(frame).items():
            readings.setdefault(m, {})[ch] = md
    chars = sorted(set(chars), key=drum.index)
    if total <= 0:
        total = (max(readings) + 1) if readings else 0
    readings = {m: readings.get(m, {}) for m in range(total)}

    shifts = {m: {ch: _shift(readings[m].get(ch), ch, drum, min_conf, spec)
                  for ch in chars} for m in range(total)}
    modes = _modes_for(drum, shifts, spec)

    # Deviant characters are re-read once; the residual uses the re-read.
    deviant = {ch for m in modes for ch in chars
               if shifts[m].get(ch) is not None
               and shifts[m][ch] != modes[m]}
    for tag, frame in frames.items():
        if not tag.startswith("p1r_"):
            continue
        frame_txt = frame.get("frame") or ""
        if not frame_txt:
            continue
        ch = frame_txt[0]
        if ch not in chars:
            continue
        for m, md in _readings_from(frame).items():
            if m in readings:
                readings[m][ch] = md

    residual: dict[int, dict[str, int | None]] = {}
    for m in modes:
        out: dict[str, int | None] = {}
        for ch in chars:
            if ch in deviant:
                out[ch] = _shift(readings[m].get(ch), ch, drum, min_conf,
                                 spec)
            else:
                out[ch] = 0 if shifts[m][ch] == modes[m] else shifts[m][ch]
        residual[m] = out

    return {
        "drum": drum,
        "chars": chars,
        "skip": sorted(skip),
        "modules": total,
        "modes": modes,
        "deviant": sorted(deviant, key=drum.index),
        "escalated": [m for m in range(total) if m not in modes],
        "residual": residual,
    }


def _run_dir(data_dir: Path, run: str | None) -> Path:
    if run:
        cand = Path(run)
        if cand.is_dir():
            return cand
        cand = data_dir / "runs" / run
        if cand.is_dir():
            return cand
        raise SystemExit(f"run not found: {run}")
    runs = sorted((data_dir / "runs").glob("run-*"))
    if not runs:
        raise SystemExit(f"no runs under {data_dir / 'runs'}")
    return runs[-1]


def _load_report(path: Path) -> dict:
    report = path / "report.json"
    if not report.is_file():
        raise SystemExit(f"no report.json in {path}")
    with open(report, encoding="utf-8") as fh:
        return json.load(fh)


def _ordered_nonzero(res: dict, chars: list[str]):
    nz = {ch: v for ch, v in res.items() if v not in (None, 0)}
    return [(ch, nz[ch]) for ch in chars if ch in nz]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", nargs="?",
                    help="run id (run-006) or run dir; default latest")
    ap.add_argument("--data-dir", default=os.environ.get("CALIB_VLM_DATA"),
                    help="data dir (default: ./data, or $CALIB_VLM_DATA)")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--list", action="store_true", help="list runs and exit")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir \
        else Path(__file__).resolve().parent / "data"
    if args.list:
        for d in sorted((data_dir / "runs").glob("run-*")):
            print(d.name)
        return 0

    run_dir = _run_dir(data_dir, args.run)
    report = _load_report(run_dir)
    recon = reconstruct(report)
    residual = {m: _ordered_nonzero(recon["residual"][m], recon["chars"])
                for m in recon["residual"]}

    if args.json:
        print(json.dumps({
            "run": run_dir.name,
            "drum": recon["drum"],
            "phases": report.get("phases"),
            "result": report.get("result"),
            "reason": report.get("reason"),
            "modes": recon["modes"],
            "escalated": recon["escalated"],
            "residual": {str(m): dict(v) for m, v in residual.items()},
        }, indent=2))
        return 0

    print(f"run {run_dir.name}: result={report.get('result')!r} "
          f"phases={report.get('phases')}")
    print(f"drum[{len(recon['drum'])}]  chars sampled={len(recon['chars'])}  "
          f"modules={recon['modules']}  p1_deviant={len(recon['deviant'])}")
    if recon["skip"]:
        print(f"skipped chars: {' '.join(repr(c) for c in recon['skip'])}")
    print("\nP1 residuals (non-zero; + = shows one flap ahead of commanded):")
    for m in range(recon["modules"]):
        if m in recon["escalated"]:
            print(f"  m{m:<3} <escalated: unreliable reads -> no residual>")
            continue
        nz = residual.get(m, [])
        mode = recon["modes"].get(m)
        body = ", ".join(f"{c!r}={v:+d}" for c, v in nz) if nz else "(none)"
        print(f"  m{m:<3} [mode {mode:+d}] {body}")
    total_nz = sum(len(v) for v in residual.values())
    print(f"\n{total_nz} non-zero residual(s) across "
          f"{len(recon['modes'])} module(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
