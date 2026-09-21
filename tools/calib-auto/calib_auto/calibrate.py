"""Auto-calibration state machine (CNN or VLM recognizer, one loop).

The loop is recognizer-agnostic: the ``reader`` it runs against is
either the local CNN (``cnn_reader.CnnReader``) or a VLM
(``reader.VlmReader``), both implementing ``read(jpeg_bytes, ...) ->
Reading``. Everything else — phases, budgets, safety rails, fleet
handling, persistence — is identical for both approaches, so a run's
result is comparable across them.

Phases (independently selectable; P0 always runs as a read-only gate):

- **P0** register: blank, all-H and index-strip frames must read back
  (camera framing/focus gate).
- **P1** coarse: reverse uniform sweep over the whole drum; the
  per-module gap-share bands give proportional module step offsets
  (whole-character multiples down to sub-character corrections).
- **P2** fine: per-character offsets from the P1 residual map, tuned
  with a parallel coarse-to-fine ladder (one cell per module per wave).
- **P4** verify: repeatability plus short forward boundary hops.
- **acceptance**: forward sweep; every module must read every
  non-space character clean.

Safety rails: motor-wear/time budgets, unreliable-read escalation
(abort when the reader cannot be trusted), confusable-glyph pairs
excluded from judgement, dry-run mode, and a
pre-run settings snapshot on disk for manual rollback.

Group 1 uses preview-then-persist; remote groups (master-only access,
no remote preview endpoint) use persist-verify-revert per cell.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections import Counter

import cv2

from .display import CalibError
from .reader import (ModuleReading, ReaderError, Reading, annotate_modules,
                     jpeg_bytes)

SUPPORTED_CONTRACT = 1
# Wear/time budgets.
MAX_FRAMES = 500
MAX_VLM_CALLS = 500
MAX_PREVIEWS = 500
MAX_PERSISTS = 500
MAX_SWEEPS = 3
# Abort when the reader cannot be trusted: more than this many
# unreliable/unreadable escalations means camera/framing/lighting is bad
# and continuing would burn wear/budget tuning noise.
MAX_UNRELIABLE_READS = 20
# P2 char-cell ladder: every per-character fault is scanned in fixed
# motor-step increments instead of one-shot exact deltas. The cap defaults
# to half a character pitch (`stepsPerChar` / 2); identity faults pass the
# firmware's ±32 char-cell clamp so the scan can reach it. A window
# narrower than the increment can fall between two candidates, so the
# half-step probes follow as a fallback chain.
P2_LADDER_STEP = 4
# Firmware limits (src/CalibApi.h + SplitFlapWebServer.cpp): char offset
# cells are motor steps clamped to ±32, and /api/calib/preview rejects a
# single |delta| > 32 with HTTP 400.
CHAR_OFFSET_LIMIT = 32
PREVIEW_DELTA_MAX = 32
# Preview-batch sizes (src/PendingActions.h + the loop-task drain in
# SplitFlapDisplay.ino): a POST /api/calib/preview-batch above
# BATCH_MAX_NUDGES is rejected with HTTP 400 (not retryable), and the
# drain copies at most REMOTE_BATCH_MAX_NUDGES nudges per remote group —
# silently dropping the rest with no error. Both caps are per request, so
# the tool chunks to them.
BATCH_MAX_NUDGES = 48
REMOTE_BATCH_MAX_NUDGES = 8
# Shift-mode sweep (P1): command every character uniformly on every
# module, walking the drum in REVERSE so each step is ~a full revolution and
# every frame passes the magnet (independently homed). The per-module
# histogram of (seen - commanded) then gives the module offset in one shot.
SWEEP_MIN_SAMPLES = 24       # trusted samples needed to judge a module
SWEEP_TRUST_PURITY = 0.80    # dominant-shift share needed to trust it
# Below the purity gate but still fixable: a single-character residual
# covering at least this share of trusted samples is a systematic fault
# fixable on the module cell, not reader noise. Escalating it would
# dead-end every affected character in P2 — the firmware's ±32 char-cell
# clamp can never hold a whole-character fix (run-005: +1 on ~68% of the
# drum, 28 "does not fit a char cell" escalations, needs-human).
SWEEP_MAJORITY_SHARE = 0.50
# Single-flap arc: at least this share of the DRUM showing the same ±1
# residual (run-005 m0: 13 of 48 characters one flap ahead while the rest
# read correct). The module is below the purity gate but is NOT reader
# noise: keep it usable (mode 0) so each arc character becomes ordinary
# per-character P2 work instead of escalating the whole module.
SWEEP_ARC_SHARE = 0.125
# P1 proportional module offset: per-direction gap-share bands over the
# full drum (48). UNIT is the floored whole-character step count; band 2
# scales it linearly so bands 1 and 2 meet exactly at BAND_FULL.
P1_STEPS_PER_CHAR = 42
P1_BAND_FULL = 0.75      # >= this share -> whole-character multiples
P1_BAND_MIN = 0.125      # >= this share -> proportional correction
P1_CONFLICT_MIN = 0.125  # losing direction at/above this -> conflicted
# Coherence floor: the top exact shift (zeros included) must hold at
# least this share of decisive samples, or the reads disagree with each
# other and the module is unjudgeable -> escalate (never band 3, so the
# MAX_UNRELIABLE_READS bad-camera rail keeps firing). 0.25 brackets the
# genuine-conflict case (19/48 = 40% must stay judged for P2) above
# reader garbage (scattered shifts, top ~5%).
P1_COHERENCE_MIN = 0.25
# Glyphs that render identically on the drum: a difference between partners
# is "no information", never evidence and never a correction.
CONFUSABLES = {
    "O": ("0", "D", "Q"), "0": ("O", "D", "Q"),
    "D": ("O", "0"), "Q": ("O", "0"),
    "I": ("1",), "1": ("I",),
    "S": ("5",), "5": ("S",),
    "Z": ("2",), "2": ("Z",),
    "B": ("8",), "8": ("B",),
    "G": ("6",), "6": ("G",),
}
# Ladder background filler rotation: non-plan modules must MOVE on every
# ladder frame. A constant filler (old: always "E") stalls a misaligned
# module — e.g. target "E" physically showing "F": the firmware resolves
# the repeat to identical step positions, commands zero steps, and every
# re-read scores the same stale flap (run-003 M0: 30/30 "E"-want/"F"-saw).
# Cycling high-contrast, non-confusable letters spread across the drum
# forces motion on every background slot, on every module, every frame.
# Seven entries (prime length): an evaluate shows targets_at + guards_at
# frames (<= 6), which never divides 7, so the filler sequence can never
# realign to the same glyphs on the next evaluate round.
LADDER_FILLERS = ("E", "M", "T", "K", "H", "W", "N")


def confusable(a: str, b: str) -> bool:
    """True when two glyphs cannot be told apart on the drum."""
    return a != b and b in CONFUSABLES.get(a, ())


def _p2_ladder_steps(steps_per_char: int,
                     cap: int | None = None) -> list[int]:
    """Signed candidate offsets for the P2 per-character ladder.

    Increments of P2_LADDER_STEP, coarse first and alternating sign, then
    the narrow-window fallback (half the increment, then one step). Each
    candidate is applied from the base, so the smallest offset that
    reads the commanded glyph wins. `cap` bounds the candidate magnitude:
    it defaults to half a character pitch (a larger correction would move
    the landing toward the neighbouring flap), while an identity fault
    passes the firmware's char-cell clamp so the scan can reach it.
    """
    if cap is None:
        cap = max(1, steps_per_char // 2)
    mags = list(range(P2_LADDER_STEP, cap + 1, P2_LADDER_STEP))
    fine: list[int] = []
    for mag in (P2_LADDER_STEP // 2, 1):
        if mag and mag <= cap and mag not in mags and mag not in fine:
            fine.append(mag)
    return ([sign * mag for mag in mags for sign in (1, -1)]
            + [sign * mag for mag in fine for sign in (1, -1)])


def _preview_chunks(delta: int, limit: int = PREVIEW_DELTA_MAX) -> list[int]:
    """Split a motor-step delta into preview calls of at most `limit` steps.

    The firmware rejects a single |delta| > 32 but applies each preview
    additively on top of the live value, so a sequence of bounded chunks
    reaches exactly the same total.
    """
    out: list[int] = []
    while delta:
        step = max(-limit, min(limit, delta))
        out.append(step)
        delta -= step
    return out


def _dominant_gap(gaps: list[int | None]) -> tuple[int, int, int] | None:
    """(sign, n, count) of the dominant gap direction.

    `n` is the minimum |gap| and `count` the same-sign sample count.
    Returns None when there is no dominant direction: empty input, an
    exact tie, or a conflicted module (losing side holds >=
    P1_CONFLICT_MIN of the drum — P2 must fix each side independently).
    Shares are over len(gaps) (one entry per swept drum character).
    """
    total = len(gaps)
    pos = [g for g in gaps if g is not None and g > 0]
    neg = [g for g in gaps if g is not None and g < 0]
    if not pos and not neg:
        return None
    if pos and neg:
        if len(pos) == len(neg):
            return None
        dom, sub = (pos, neg) if len(pos) > len(neg) else (neg, pos)
        if total and len(sub) / total >= P1_CONFLICT_MIN:
            return None
    else:
        dom = pos or neg
    sign = 1 if dom is pos else -1
    return (sign, min(abs(g) for g in dom), len(dom))


def p1_module_steps(gaps: list[int | None],
                    unit: int = P1_STEPS_PER_CHAR) -> tuple[int, int]:
    """Proportional P1 module offset from per-character gap counts.

    `gaps` is one signed shift per drum character (`None` = excluded
    sample: untrusted read, confusable pair, blank-flap misfire). The
    dominant share `x` is over the FULL drum (len(gaps)) — excluded
    samples dilute, never concentrate. `unit` is the floored
    whole-character step count for THIS drum (floor(steps_per_rot /
    len(drum)): 42 on a 48-drum, 55 on a 37-drum). Returns (signed motor
    steps, band 1|2|3); band 3 means "judged, move nothing" (residuals go
    to P2).
    """
    if not gaps:
        return (0, 3)
    dom = _dominant_gap(gaps)
    if dom is None:
        return (0, 3)
    sign, n, count = dom
    x = count / len(gaps)
    if x >= P1_BAND_FULL:
        return (sign * n * unit, 1)
    if x >= P1_BAND_MIN:
        # Half-up rounding (not banker's round): floor(v + 0.5).
        return (sign * int(unit * n * x / P1_BAND_FULL + 0.5), 2)
    return (0, 3)


def _parse_csv_matrix(raw, rows: int, cols: int) -> list[list[int]]:
    """Parse a settings matrix (list of lists or "a,b;c,d" string)."""
    out = [[0] * cols for _ in range(rows)]
    if isinstance(raw, list):
        for r, row in enumerate(raw[:rows]):
            for c, value in enumerate(list(row)[:cols]):
                try:
                    out[r][c] = int(value)
                except (TypeError, ValueError):
                    pass
        return out
    r = 0
    for part in str(raw or "").split(";"):
        if r >= rows:
            break
        values = [v.strip() for v in part.split(",") if v.strip() != ""]
        for c, value in enumerate(values[:cols]):
            try:
                out[r][c] = int(value)
            except ValueError:
                pass
        r += 1
    return out


class _LoggingDisplay:
    """Proxy that logs mutating display API calls as `api` events.

    Read-only polls (`status`, `wait_settled`) are deliberately NOT
    logged: `wait_settled` polls `status()` every 0.5 s and would flood
    the log. Everything mutating (show/preview/persist/reload/hold)
    plus one-shot reads (snapshot/contract) is logged with params and
    result so the run log shows every display API call.
    """

    # Mutating calls wrapped with an `api` log line. Read-only polls
    # (`status`, `wait_settled`) are NOT wrapped: `wait_settled` polls
    # every 0.5 s and would flood the log.
    _LOGGED = ("show", "show_and_settle", "preview", "preview_batch",
               "persist", "reload", "hold", "snapshot", "contract",
               "wait_settled")

    def __init__(self, display, event_fn):
        object.__setattr__(self, "_display", display)
        object.__setattr__(self, "_event", event_fn)

    def __getattr__(self, name):
        display = object.__getattribute__(self, "_display")
        target = getattr(display, name)  # raises AttributeError if missing
        if name not in self._LOGGED or not callable(target):
            return target
        event_fn = object.__getattribute__(self, "_event")

        def wrapper(*args, **kwargs):
            t0 = time.monotonic()
            out = target(*args, **kwargs)
            dt = time.monotonic() - t0
            try:
                event_fn("api", self._describe(name, args, kwargs, out)
                         + f" in {dt:.1f}s")
            except Exception:
                pass  # logging must never break a run
            return out

        return wrapper

    @staticmethod
    def _describe(name, args, kwargs, out):
        if name == "show_and_settle":
            frame = args[0] if args else kwargs.get("frame", "?")
            dwell = args[1] if len(args) > 1 else kwargs.get("dwell_ms", 800)
            fid = out.get("frameId") if isinstance(out, dict) else "?"
            return (f"POST show frame={frame!r} dwellMs={dwell} "
                    f"-> frameId {fid} settled")
        if name == "show":
            frame = args[0] if args else kwargs.get("frame", "?")
            return f"POST show frame={frame!r}"
        if name == "preview":
            module, ci = args[0], args[1] if len(args) > 1 else "?"
            delta = args[2] if len(args) > 2 else kwargs.get("delta", "?")
            return (f"POST preview module={module} charIndex={ci} "
                    f"delta={delta:+d}" if isinstance(delta, int)
                    else f"POST preview module={module} charIndex={ci}")
        if name == "preview_batch":
            items = list(args[0]) if args else []
            summary = ", ".join(
                f"m{n.get('module')} c{n.get('charIndex')}:{n.get('delta'):+d}"
                for n in items[:8])
            more = f" +{len(items) - 8} more" if len(items) > 8 else ""
            return f"POST preview-batch {len(items)} nudge(s): {summary}{more}"
        if name == "persist":
            scope = args[0] if args else kwargs.get("scope", "?")
            kind = args[1] if len(args) > 1 else kwargs.get("kind", "?")
            value = args[2] if len(args) > 2 else kwargs.get("value", "?")
            return f"POST offsets scope={scope} kind={kind} value={value}"
        if name == "reload":
            return "POST reload -> volatile previews reverted"
        if name == "hold":
            active = args[0] if args else kwargs.get("active", "?")
            return f"POST hold active={bool(active)}"
        if name == "wait_settled":
            return "wait settled (homing/poll)"
        if name == "snapshot":
            return "GET settings -> snapshot saved"
        if name == "contract":
            ver = out.get("contractVersion") if isinstance(out, dict) else "?"
            return f"GET calib-contract -> version {ver}"
        return f"{name} called"


class Calibrator:
    def __init__(self, display, camera, reader, photo_dir: str,
                 dwell_ms: int = 800, timeout_s: float = 60.0,
                 min_confidence: float = 0.6, exhaustive: bool = False,
                 mode: str = "full", on_event=None, max_seconds: float = 3600.0,
                  phases: list[str] | None = None,
                  run_context: dict | None = None):
        if mode not in ("dry-run", "full"):
            raise ValueError("mode must be dry-run or full")
        self.display = display
        self.camera = camera
        self.reader = reader
        self.photo_dir = photo_dir
        self.dwell_ms = dwell_ms
        self.timeout_s = timeout_s
        self.min_confidence = min_confidence
        self.exhaustive = exhaustive
        self.mode = mode
        # Independently selectable phases (P0 always runs as a read-only
        # safety gate). Per-run only; None/empty = full chain.
        self.phases = self._normalize_phases(phases)
        self.on_event = on_event or (lambda e: None)
        self.max_seconds = max_seconds
        # Exhaustive mode visits every residue class so each drum
        # character lands on each module at least once (slower/wear).
        self.max_sweeps = MAX_SWEEPS if not exhaustive else 8
        self.max_previews = MAX_PREVIEWS if not exhaustive else MAX_PREVIEWS * 4
        self.max_persists = MAX_PERSISTS if not exhaustive else MAX_PERSISTS * 4
        self.max_frames = MAX_FRAMES if not exhaustive else MAX_FRAMES * 4
        self.max_vlm_calls = MAX_VLM_CALLS if not exhaustive else MAX_VLM_CALLS * 4
        self.frames: list[dict] = []
        self.deltas: list[dict] = []
        self.identity_persistent: list[dict] = []
        # Modules fixed at module level (P1 identity or alignment): P2/P3
        # treat their per-char suspects as unverified until a re-check of
        # the suspect frame still shows the fault (a stale module offset
        # moves every character on the drum, so tuning char cells on top
        # of it would bake the error in twice).
        self.module_fixed: set[int] = set()
        self.overlay: dict[tuple[int, int, int], int] = {}
        self.residue: dict[tuple[int, int, int], int] = {}
        self.old_values: dict[tuple[int, int, int], int] = {}
        self.previews = 0
        self.persists = 0
        self.sweeps = 0
        self.frames_used = 0
        self.vlm_calls = 0
        self.aborted = False
        self.report: dict | None = None
        self.total = 0
        self.charset = 0
        self.drum = ""
        self._last_shown_frame: str | None = None
        self._last_frame_id: int = 0
        # Motor steps between two drum characters (stepsPerRot / drum
        # length), read from /settings. Identity fixes move a whole
        # character (delta * steps_per_char).
        self.steps_per_char = 1
        self.group_widths = [0]
        # Remote offsets from the master's /settings snapshot (rModOffs,
        # rChrOff0..4): the only way to know the absolute base for a
        # remote-cell persist, since status exposes local cells only.
        self.remote_mod: list[list[int]] = []
        self.remote_char: list[list[list[int]]] = []
        self._reader_failures = 0
        self._unreliable_count = 0
        self._phase_marks: list[dict] = []
        self._vlm_tokens = {"prompt_tokens": 0, "completion_tokens": 0}
        self._vlm_seconds = 0.0
        self._wall_start = 0.0
        # Reproducibility context supplied by the server (camera/reader
        # settings, model; never secrets). Tuner params are added in run().
        self.run_context = dict(run_context or {})
        self._t0 = time.monotonic()
        os.makedirs(photo_dir, exist_ok=True)
        # Log every mutating display API call as `api` events so the run
        # log shows the full display traffic (status polls stay unlogged).
        self.display = _LoggingDisplay(display, self.event)

    @staticmethod
    def _normalize_phases(phases) -> list[str]:
        """Validate a per-run phase selection; None/empty = full chain."""
        order = ("p1", "p2", "p4", "acceptance")
        if phases in (None, ""):
            return list(order)
        if isinstance(phases, str):
            phases = [phases]
        try:
            items = [str(p).strip().lower() for p in list(phases)]
        except TypeError:
            raise ValueError("phases must be a list of phase names")
        unknown = [p for p in items if p not in order]
        if unknown:
            raise ValueError(f"unknown phases: {', '.join(unknown)} "
                             f"(choose from {', '.join(order)})")
        # De-duplicate, keep canonical order.
        return [p for p in order if p in items] or list(order)

    # -- fallback glyph -----------------------------------------------------
    def _pick_fallback(self, avoid: set[str]) -> str:
        """A drum glyph outside `avoid` (never blank)."""
        for cand in ("E", "H", "M", "T", "K"):
            if cand in self.drum and cand not in avoid:
                return cand
        for cand in self.drum:
            if cand != " " and cand not in avoid:
                return cand
        return self.drum[1] if len(self.drum) > 1 else self.drum[0]

    # -- events / abort -------------------------------------------------------
    def event(self, kind: str, text: str, photo: str | None = None,
              detail: dict | None = None):
        evt: dict = {"t": time.strftime("%H:%M:%S"),
                     "elapsed": round(time.monotonic() - self._t0, 2),
                     "kind": kind, "text": text, "photo": photo}
        if detail is not None:
            evt["detail"] = detail
        if kind == "phase":
            # Phase boundary with budget snapshot: post-run analysis can
            # derive per-phase seconds + frames/read/preview/persist cost
            # (`_phase_timing` books each interval to the phase announced
            # by the PREVIOUS mark).
            self._phase_marks.append({
                "name": text, "elapsed": evt["elapsed"],
                "frames": self.frames_used, "vlmCalls": self.vlm_calls,
                "previews": self.previews, "persists": self.persists})
        self.on_event(evt)

    def _abort_requested(self) -> bool:
        return self.aborted

    def abort(self):
        self.aborted = True

    def _guard_budgets(self, frame: bool = False, preview: bool = False):
        if self.aborted:
            raise CalibError("aborted by user")
        if frame:
            if self.frames_used >= self.max_frames:
                raise CalibError("frame budget exhausted")
            self.frames_used += 1
            if self.vlm_calls >= self.max_vlm_calls:
                raise CalibError("reader call budget exhausted")
        if preview and self.previews >= self.max_previews:
            raise CalibError("preview budget exhausted")
        if self.persists >= self.max_persists:
            raise CalibError("persist budget exhausted")
        if self.sweeps >= self.max_sweeps:
            raise CalibError("sweep budget exhausted")
        if time.monotonic() - self._t0 > self.max_seconds:
            raise CalibError("time budget exhausted")

    def _charge_vlm(self, calls: int) -> None:
        """Account recognizer round trips after a frame is read."""
        self.vlm_calls += calls
        if self.vlm_calls > self.max_vlm_calls:
            raise CalibError("reader call budget exhausted")

    # -- display helpers (duck-type tolerant, abort-responsive) ---------------
    def _wait_settled(self, timeout_s: float) -> dict:
        try:
            return self.display.wait_settled(
                timeout_s, abort_flag=self._abort_requested)
        except TypeError:
            if self.aborted:
                raise CalibError("aborted by user")
            out = self.display.wait_settled(timeout_s)
            if self.aborted:
                raise CalibError("aborted by user")
            return out

    def _show_and_settle(self, frame: str) -> dict:
        try:
            return self.display.show_and_settle(
                frame, self.dwell_ms, self.timeout_s,
                abort_flag=self._abort_requested)
        except TypeError:
            if self.aborted:
                raise CalibError("aborted by user")
            out = self.display.show_and_settle(frame, self.dwell_ms,
                                               self.timeout_s)
            if self.aborted:
                raise CalibError("aborted by user")
            return out

    def _clear_volatile_previews(self):
        """Revert RAM-only previews to the persisted offsets.

        A previous run (or a dry-run/preview run) can leave uncommitted
        preview nudges live on the controller. The settings rollback does
        NOT clear them (it only reloads when a calibration value changes),
        so the next run would read those ghost offsets as the baseline and
        chase corrections that do not exist. Older firmware has no reload
        endpoint: skip best-effort and warn.
        """
        reload_fn = getattr(self.display, "reload", None)
        if reload_fn is None:
            return
        try:
            reload_fn()
            self._wait_settled(self.timeout_s)
            self.event("phase", "reverted volatile previews to persisted "
                                "offsets")
        except CalibError as exc:
            # Distinguish a genuinely missing endpoint (older firmware,
            # HTTP 404/405) from a transient failure (busy 409, network):
            # the old message blamed the endpoint for every error.
            text = str(exc)
            if "HTTP 404" in text or "HTTP 405" in text:
                self.event("error", "reload endpoint unavailable; baseline "
                                    "may include uncommitted previews")
            else:
                self.event("error", f"preview revert failed ({text}); "
                                    "baseline may include uncommitted "
                                    "previews")

    # -- reading one frame ----------------------------------------------------
    def _summary_line(self, reading: Reading) -> str:
        counts: dict[str, int] = {}
        for m in reading.modules:
            key = m.condition
            counts[key] = counts.get(key, 0) + 1
        return ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))

    @staticmethod
    def _read_line(frame: str, reading: Reading) -> str:
        """One-line expected-vs-read verdict with per-module mismatches.

        `want` is the commanded frame, `saw` the reader's transcription;
        every module where they differ (or the flap is not clean) is
        listed as `m<i> want 'X' saw 'Y'/cond` so a glance at the log
        shows which modules are off and how.
        """
        want = frame.ljust(len(reading.modules))[:len(reading.modules)]
        saw = reading.text
        bad = []
        for i, m in enumerate(reading.modules):
            exp = want[i] if i < len(want) else " "
            if m.char != exp or m.condition not in ("clean", "blank"):
                bad.append(f"m{i} want {exp!r} saw {m.char!r}/{m.condition}")
        line = f"want {want!r} saw {saw!r} ({len(bad)}/{len(reading.modules)} off)"
        if bad:
            line += " — " + ", ".join(bad)
        return line

    def _show_read(self, frame: str, tag: str) -> tuple[dict, Reading]:
        """Show a frame, settle, photograph and read it."""
        self._guard_budgets(frame=True)
        info = self._show_and_settle(frame)
        self._last_shown_frame: str | None = frame
        self._last_frame_id: int = info["frameId"]
        return self._capture_read(frame, tag, info["frameId"])

    def _reread(self, frame: str, tag: str) -> tuple[dict, Reading]:
        """Settle, photograph and read WITHOUT re-showing the frame.

        Re-showing an identical frame from just past its target drives the
        drum forward almost a full revolution to come back to it — every
        such revolution costs seconds and accumulates homing error, so a
        tune loop that re-shows the same frame walks the module further
        off each iteration. After a preview the drum is already at the
        frame: settle + capture is the correct re-read.
        """
        self._guard_budgets(frame=True)
        self._wait_settled(self.timeout_s)
        info_id = self._last_frame_id
        return self._capture_read(frame, f"{tag}_r", info_id)

    def _capture_read(self, frame: str, tag: str,
                      frame_id: int) -> tuple[dict, Reading]:
        if self.dwell_ms:
            time.sleep(self.dwell_ms / 1000.0)
        # Drain stale buffered frames first: the first grab after a show
        # can be a frame exposed BEFORE the move (one-frame lag).
        capture = getattr(self.camera, "capture_fresh",
                          self.camera.capture)
        img = capture()
        path = os.path.join(self.photo_dir, f"{tag}_f{frame_id}.png")
        if not cv2.imwrite(path, img):
            raise CalibError(f"failed to write photo {path}")
        send = annotate_modules(img, self.total) if self.reader.annotate else img
        vlm = getattr(self.reader, "vlm", None)
        model = (getattr(vlm, "model", None)
                 or getattr(self.reader, "backend", None) or "local")
        t0 = time.monotonic()
        try:
            reading = self.reader.read(jpeg_bytes(send), total=self.total,
                                       expected=frame, charset=self.drum,
                                       drum=self.drum)
            note = getattr(reading, "unreliable_note", "") or ""
            if getattr(reading, "unreliable", False):
                # The reader knows its own geometry failed (display not
                # located, module grid not on the modules): its characters
                # are guesses, so nothing may be tuned on them. Count it
                # like a reader failure — three in a row abort with the
                # reader's own note instead of burning a whole sweep and
                # then escalating module by module.
                self._reader_failures += 1
                self.event("error", f"reader unreliable on {tag}: {note}")
                if self._reader_failures >= 3:
                    raise CalibError(f"reader unreliable: {note}")
            else:
                self._reader_failures = 0
        except ReaderError as exc:
            self._reader_failures += 1
            reading = self.reader.error_reading(self.total, frame, str(exc))
            self.event("error", f"reader failed on {tag}: {exc}")
            if self._reader_failures >= 3:
                raise CalibError(
                    f"reader failing repeatedly: {exc}") from exc
        calls = max(1, getattr(self.reader, "last_calls", 1))
        dt = time.monotonic() - t0
        self._vlm_seconds += dt
        usage = getattr(self.reader, "last_usage", None) or {}
        try:
            self._vlm_tokens["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
            self._vlm_tokens["completion_tokens"] += int(usage.get("completion_tokens", 0))
        except (TypeError, ValueError):
            pass
        tok = ""
        if usage.get("prompt_tokens") or usage.get("completion_tokens"):
            tok = (f" tok {usage.get('prompt_tokens', 0)}/"
                   f"{usage.get('completion_tokens', 0)}")
        self.event("vlm", f"read model={model} tag={tag} "
                           f"->{calls} call(s) in {dt:.1f}s{tok}: "
                           f"{self._read_line(frame, reading)}")
        # Account the real recognizer round trips: a VLM read may re-ask
        # once, and failed parses still consumed provider calls.
        self._charge_vlm(calls)
        rec = {"tag": tag, "frameId": frame_id, "frame": frame,
               "photo": os.path.basename(path)}
        rec.update(reading.as_dict())
        boxes = getattr(self.reader, "last_boxes", None)
        if boxes is not None:
            rec["boxes"] = [list(box) for box in boxes]
            rec["display"] = getattr(self.reader, "last_display", None)
        self.frames.append(rec)
        self.event("read", f"{tag}: {self._read_line(frame, reading)} "
                           f"({self._summary_line(reading)})",
                   rec["photo"], detail=rec)
        return rec, reading

    # -- fleet geometry -------------------------------------------------------
    @staticmethod
    def _parse_widths(raw, groups: int) -> list[int] | None:
        """First `groups` per-group module counts of a CSV/list, or None.

        `masterGroupModuleCounts` is a user-editable CSV (firmware default
        "8,8,8,8,8,8"); `GET /api/calib/status` reports the same data as a
        `groupWidths` list. Entry 0 is group 1 (the local controller),
        entry 1 group 2, ... (SplitFlapEspNow::getGroupModuleCount).
        """
        if raw is None or raw == "":
            return None
        items = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")
        try:
            widths = [int(str(part).strip()) for part in items
                      if str(part).strip() != ""]
        except (TypeError, ValueError):
            return None
        if len(widths) < groups or any(w <= 0 for w in widths[:groups]):
            return None
        return widths[:groups]

    def _declared_widths(self, status: dict, settings: dict,
                         total: int, local: int, groups: int) -> list[int] | None:
        """Group widths the firmware declares, or None when it declares none.

        The firmware maps fleet modules through `masterGroupModuleCounts`,
        so assuming every group is local-wide mis-maps every module past
        the first group boundary. Prefer the status endpoint's
        `groupWidths` (list, group 1 first); fall back to the
        `masterGroupModuleCounts` CSV of the /settings snapshot. A
        `groupWidths` that is present but contradicts the status counts is
        an error — guessing a mapping would silently mis-tune.
        """
        if status.get("groupWidths") is not None:
            raw = status["groupWidths"]
            widths = self._parse_widths(raw, groups)
            if (widths is None or len(widths) != groups
                    or widths[0] != local or sum(widths) != total):
                raise CalibError(
                    f"fleet geometry inconsistent: groupWidths {raw!r} "
                    f"does not cover {total} total / {local} local module(s) "
                    f"over {groups} group(s)")
            return widths
        widths = self._parse_widths(settings.get("masterGroupModuleCounts"),
                                    groups)
        if widths is not None and widths[0] == local and sum(widths) == total:
            return widths
        return None

    def _widths(self, status: dict, settings: dict | None = None) -> list[int]:
        """Module count per group, group 1 (local) first.

        Order of preference: the firmware's declared group widths (status
        `groupWidths`, else the `masterGroupModuleCounts` CSV the run
        already fetches), then the legacy equal-width heuristic for
        firmware that reports neither.
        """
        total = int(status["totalModules"])
        local = int(status["numModules"])
        groups = int(status.get("groupCount", 1)) or 1
        declared = self._declared_widths(status, settings or {}, total,
                                         local, groups)
        if declared is not None:
            return declared
        if groups <= 1:
            if total != local:
                raise CalibError(
                    f"fleet geometry inconsistent: totalModules {total} but "
                    f"groupCount {groups} covers only {local} local modules")
            return [local]
        widths = [local] * groups
        widths[-1] = total - local * (groups - 1)
        return widths

    def _group_of(self, module: int) -> int:
        off = 0
        for g, width in enumerate(self.group_widths, start=1):
            if module < off + width:
                return g
            off += width
        raise CalibError(f"module {module} outside fleet geometry "
                         f"{self.group_widths}")

    def _local_index(self, module: int) -> int:
        off = sum(self.group_widths[: self._group_of(module) - 1])
        return module - off

    def _read_cell(self, group: int, local: int, char_index: int) -> int:
        st = self.display.status()
        if group == 1:
            if char_index < 0:
                mods = st.get("moduleOffsets", [])
                return int(mods[local]) if local < len(mods) else 0
            rows = st.get("charOffsets", [])
            if local < len(rows):
                try:
                    row = rows[local]
                    if char_index < len(row):
                        return int(row[char_index])
                except (TypeError, ValueError):
                    pass
            return 0
        row = group - 2
        if 0 <= row < len(self.remote_mod):
            if char_index < 0:
                mods = self.remote_mod[row]
                return int(mods[local]) if local < len(mods) else 0
            if row < len(self.remote_char) and local < len(self.remote_char[row]):
                cells = self.remote_char[row][local]
                if char_index < len(cells):
                    return int(cells[char_index])
        return 0

    def _load_remote_offsets(self, settings: dict):
        self.remote_mod = _parse_csv_matrix(settings.get("rModOffs"), 5, 8)
        self.remote_char = [_parse_csv_matrix(settings.get(f"rChrOff{row}"),
                                             8, 48)
                            for row in range(5)]

    def _ensure_cell(self, group: int, local: int, char_index: int):
        key = (group, local, char_index)
        if key not in self.overlay:
            self.overlay[key] = self._read_cell(group, local, char_index)
            self.old_values[key] = self.overlay[key]
            self.residue[key] = 0
        return key

    def live(self, key) -> int:
        return self.overlay.get(key, 0) + self.residue.get(key, 0)

    @staticmethod
    def _kind(char_index: int) -> str:
        return "char" if char_index >= 0 else "module"

    # -- tuning primitives ----------------------------------------------------
    def _trusted(self, entry: ModuleReading) -> bool:
        return (entry.source in ("vlm", "classifier")
                and entry.char is not None and entry.char != ""
                # '?' is the unknown sentinel, but it is also a real drum
                # character: trust it when that is what was commanded.
                and (entry.char != "?" or entry.expected == "?")
                and entry.condition != "unreadable"
                and entry.confidence >= self.min_confidence)

    def _drum_delta(self, seen: str, target: str) -> int | None:
        """Signed-minimal character distance from `seen` to `target`.

        An offset correction re-anchors where the firmware thinks each
        character sits (then re-homes) — it is NOT a drum move, so the
        drum's forward-only rule does not apply. The minimal signed
        delta reaches the identical physical position: one char ahead
        is -1 char, not a near-full revolution forward (which ground
        the drum through dozens of chunked previews for a 1-char fix).
        """
        if not self.drum or seen not in self.drum or target not in self.drum:
            return None
        n = len(self.drum)
        raw = (self.drum.index(target) - self.drum.index(seen)) % n
        return raw - n if raw > n // 2 else raw

    def _escalate(self, module: int, glyph: str, note: str):
        rec = {"module": module, "glyph": glyph, "note": note}
        if rec not in self.identity_persistent:
            self.identity_persistent.append(rec)
        self.event("error", f"escalate m{module} target {glyph!r}: {note}")
        lowered = note.lower()
        if "unreliable" in lowered or "unreadable" in lowered:
            self._unreliable_count += 1
            if self._unreliable_count > MAX_UNRELIABLE_READS:
                raise CalibError(
                    f"too many unreliable reads "
                    f"({self._unreliable_count} > {MAX_UNRELIABLE_READS}); "
                    f"aborting - check camera/framing/lighting")

    def _record_delta(self, module: int, char_index: int, target: str,
                      before: ModuleReading, after: ModuleReading,
                      fixed: bool, proposal: bool = False,
                      delta: int | None = None) -> dict:
        group = self._group_of(module)
        local = self._local_index(module)
        key = (group, local, char_index)
        entry = {"scope": group, "module": local, "globalModule": module,
                 "charIndex": char_index, "kind": self._kind(char_index),
                 "target": target, "proposal": proposal, "fixed": fixed,
                 "old": self.old_values.get(key),
                 "new": self.overlay.get(key),
                 "residue": self.residue.get(key),
                 "before": before.as_dict() if before else None,
                 "after": after.as_dict() if after else None}
        if delta is not None:
            entry["delta"] = delta
        self.deltas.append(entry)
        state = "fixed" if fixed else ("proposed" if proposal else "unresolved")
        before_text = f"{before.char!r}/{before.condition}" if before else "?"
        after_text = f"{after.char!r}/{after.condition}" if after else "?"
        self.event("tune", f"{state} g{group} m{local} c{char_index} "
                           f"target {target!r}: {before_text} -> {after_text}",
                   None, detail=entry)
        return entry

    def _commit_local(self, group: int, key, kind: str, char_index: int):
        """Persist a verified local preview residue as an absolute value."""
        if self.mode != "full" or group != 1 or self.residue.get(key, 0) == 0:
            return
        self._guard_budgets()
        self.display.persist(group, kind, self.live(key), key[1],
                             max(char_index, 0))
        self.persists += 1
        self.overlay[key] = self.live(key)
        self.residue[key] = 0
        self._wait_settled(self.timeout_s)

    # -- phases ---------------------------------------------------------------
    def _index_strip(self) -> str:
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return "".join(chars[i % len(chars)] for i in range(self.total))

    @staticmethod
    def _visible(text: str) -> str:
        return "".join(ch for ch in text if ch != " ")

    def _looks_mirrored(self, frame: str, reading: Reading) -> bool:
        read = self._visible(reading.text)
        expected = self._visible(frame)
        return (len(read) >= 4 and read == expected[::-1]
                and read != expected)

    def _p0_register(self):
        self.event("phase", "P0 register: blank, all-H, index strip")
        problems = []
        blank = self._show_read(" " * self.total, "p0_blank")[1]
        all_h = self._show_read("H" * self.total, "p0_allH")[1]
        frame = self._index_strip()
        _, idx = self._show_read(frame, "p0_index")
        # Blank frames legitimately collapse in a reader answer (all
        # positions look identical); reconciliation pads them. The index
        # strip is the strict one: every position carries a distinct glyph.
        for tag, reading in (("blank", blank), ("all-H", all_h)):
            if reading.realigned:
                self.event("read", f"P0 {tag}: reader answer realigned "
                                   f"({reading.raw_count}/{self.total} entries)")
        if idx.realigned and idx.raw_count < self.total:
            problems.append(f"index strip: reader returned only "
                            f"{idx.raw_count} of {self.total} entries")
        if all(m.char in (" ", "?") for m in idx.modules):
            problems.append("index strip unreadable (camera framing/focus?)")
        if self._looks_mirrored(frame, idx):
            problems.append("read order looks mirrored; fix camera orientation")
        if problems:
            raise CalibError("P0 registration failed: " + "; ".join(problems))

    def _batch_nudge(self, nudges: list[tuple[int, int, int, int]]):
        """Apply cell nudges [(group, local, char_index, delta), ...].

        Every nudge is one entry of the `nudges` array of
        `POST /api/calib/preview-batch`, shaped
        {"scope": <1..6>, "module": <module index inside that scope>,
         "charIndex": <drum index, -1 = the module cell>,
         "delta": <motor steps, non-zero, |delta| <= stepsPerRot>}.
        Scope 1 is the local controller (applied straight away); scopes
        2..6 are remote groups, forwarded by the master over ESP-NOW and
        applied RAM-only there (the master's busy fence covers their
        homing via preview acks). N cells across the fleet cost one homing
        pass per chunk, not N.

        Firmware caps, which this method must respect itself:
        - at most BATCH_MAX_NUDGES (48) nudges per call — the endpoint
          rejects more with HTTP 400, which is not retryable;
        - at most REMOTE_BATCH_MAX_NUDGES (8) nudges per remote group per
          call — the drain forwards only the first 8 and silently drops
          the rest, so an oversized remote slice would be lost.
        The nudges are therefore sent in order, chunked per scope (<=48
        for the local scope, <=8 for each remote scope).

        Char cells accept |delta| <= 32 per entry but apply additively, so
        a larger correction is split into chunks inside the same pass.
        `self.overlay`/`self.residue` track the device's live value for
        every touched cell after each chunk (residue mirroring the
        firmware's ±32 char-cell clamp) and `self.previews` is charged
        once per nudge sent. Falls back to serial single previews on older
        firmware (local group only).
        """
        if not nudges:
            return
        # Char cells reject |delta| > 32 per call but apply additively, so
        # split them into bounded chunks inside the same batch pass.
        expanded: list[tuple[int, int, int, int]] = []
        for group, local, char_index, delta in nudges:
            if char_index >= 0 and abs(delta) > PREVIEW_DELTA_MAX:
                for chunk in _preview_chunks(delta):
                    expanded.append((group, local, char_index, chunk))
            else:
                expanded.append((group, local, char_index, delta))
        nudges = expanded
        if self.previews + len(nudges) > self.max_previews:
            raise CalibError("preview budget exhausted")
        self._guard_budgets(preview=True)
        batch = getattr(self.display, "preview_batch", None)
        by_scope: dict[int, list[tuple[int, int, int]]] = {}
        for group, local, char_index, delta in nudges:
            by_scope.setdefault(group, []).append((local, char_index, delta))
        for group, items in sorted(by_scope.items()):
            applied = False
            if batch is not None:
                limit = (BATCH_MAX_NUDGES if group == 1
                         else REMOTE_BATCH_MAX_NUDGES)
                for start in range(0, len(items), limit):
                    chunk = items[start:start + limit]
                    batch([{"scope": group, "module": local,
                            "charIndex": char_index, "delta": delta}
                           for local, char_index, delta in chunk])
                applied = True
            elif group == 1:
                for local, char_index, delta in items:
                    self.display.preview(local, char_index, delta)
                applied = True
            else:
                self.event("error",
                           f"remote preview needs the fleet endpoint; "
                           f"skipping {len(items)} nudge(s) on group {group}")
                continue  # remote preview needs the fleet endpoint
            if applied:
                for local, char_index, delta in items:
                    key = (group, local, char_index)
                    if key not in self.overlay:
                        self._ensure_cell(group, local, char_index)
                    residue = self.residue.get(key, 0) + delta
                    if char_index >= 0:
                        # Mirror the firmware: a char preview clamps the cell's
                        # ABSOLUTE value to ±32, so must the tracked belief.
                        absolute = self.overlay.get(key, 0) + residue
                        residue = (max(-CHAR_OFFSET_LIMIT,
                                       min(CHAR_OFFSET_LIMIT, absolute))
                                   - self.overlay.get(key, 0))
                    self.residue[key] = residue
        self.previews += len(nudges)
        self._wait_settled(self.timeout_s)

    def _cell_ladder(self, plans: list[dict]) -> set[int]:
        """Parallel coarse-to-fine ladder over independent cell offsets.

        Plans are cells (module offsets or per-character cells). Every plan
        applies its own candidate in the SAME mixed frames/reads and the
        batch preview homes the touched modules in one pass, so the cost is
        `steps x frames`, not `plans x steps x frames`. Callers pass at
        most ONE plan per module (a frame commands one character per
        module); same-module cells are tuned in separate calls. A character
        plan scores only whether its target glyph reads correctly (this
        hardware cannot show a half- or double-seated flap); module plans
        keep the conservative correct+clean-minus-guard-breakage score. A
        candidate is kept only when it raises the plan's score, so the
        smallest clearing shift wins. A cell that reaches a perfect score
        stops being probed (later candidates can only tie it), and an
        exhausted cell holds at its best offset instead of snapping back
        toward the base every remaining round.
        """
        if not plans or self.mode == "dry-run":
            return set()
        if len({p["module"] for p in plans}) != len(plans):
            # A frame commands exactly one character per module and scores
            # are keyed by module, so two plans on one module would share a
            # score and a frame slot: both could "win" from one verified
            # reading and commit unverified offsets. Callers split
            # same-module cells into waves.
            raise ValueError("_cell_ladder needs one plan per module; "
                             "split same-module cells into separate calls")
        steps_at = max(len(p["steps"]) for p in plans)
        # Seed every cell's base BEFORE nudging: the applied candidates must
        # land in the residue so a commit persists overlay + residue.
        for p in plans:
            key = self._ensure_cell(p["group"], p["local"], p["char_index"])
            p["cap_base"] = self.live(key)
            p.setdefault("cap_abs", p["char_index"] >= 0)
        by_module = {p["module"]: p for p in plans}
        targets_at = max(len(p["targets"]) for p in plans)
        guards_at = max(len(p["guards"]) for p in plans)
        frames_per_eval = targets_at + guards_at
        # Monotonic ladder frame counter: filler rotation keys off it (one
        # bump per shown frame). Any background/exhausted slot must show a
        # DIFFERENT glyph every frame AND every evaluate round: a repeat
        # resolves to identical step positions, the firmware commands zero
        # steps, and every re-read scores the same stale flap (run-003
        # M0: 30/30 "E"-want/"F"-saw on a constant "E" filler). The
        # last_filler guard keeps adjacent frames distinct even when the
        # cycle length happens to realign with frames_per_eval.
        seq = [0]
        last_filler = [None]

        def filler_for(key: int) -> str:
            pool = [c for c in LADDER_FILLERS if c in self.drum]
            if len(pool) >= 2:
                start = key % len(pool)
                for cand in pool[start:] + pool[:start]:
                    if cand != last_filler[0]:
                        last_filler[0] = cand
                        return cand
            if pool:
                last_filler[0] = pool[0]
                return pool[0]
            return self._pick_fallback(set())

        def frame_for(index: int) -> str:
            # Filler key is the monotonic seq alone (bumped once per shown
            # frame): consecutive frames ALWAYS differ on background slots,
            # including across evaluate boundaries.
            filler = filler_for(seq[0])
            cells = []
            for module in range(self.total):
                p = by_module.get(module)
                if p is None:
                    cells.append(filler)
                elif index < len(p["targets"]):
                    cells.append(p["targets"][index])
                elif index - targets_at < len(p["guards"]):
                    cells.append(p["guards"][index - targets_at])
                elif len(p["guards"]) > 1:
                    # Exhausted plan slot: rotate BY EVALUATE ROUND, not
                    # by seq — seq advances frames_per_eval per evaluate,
                    # which can be a multiple of len(guards) and would
                    # realign to the same glyph every round.
                    r = (seq[0] - 1) // max(1, frames_per_eval)
                    slot = index - targets_at
                    cells.append(p["guards"][(slot + r) % len(p["guards"])])
                else:
                    cells.append(filler)
            return "".join(cells)

        def evaluate(moved: set[int] | None = None,
                     cache: dict | None = None) -> dict[int, int]:
            # A plan cell whose state did not change keeps the offset it
            # was already scored under, so its score cannot have changed
            # when its shown glyph sequence is identical to the cached
            # one: reuse that score instead of re-reading a stale flap.
            # Sequence check matters because exhausted-guard slots rotate
            # with seq — there the glyph (and the drum) DID change, so
            # the score must be taken fresh. When NOTHING moved, skip
            # the re-show entirely.
            if cache is not None and moved is not None and not moved:
                return dict(cache["scores"])
            # One seq bump per frame: the filler must rotate between
            # consecutive frames WITHIN an evaluate, not just across
            # evaluates (background modules must move every frame).
            frames = []
            for index in range(targets_at + guards_at):
                seq[0] += 1
                frames.append(frame_for(index))
            readings = []
            for index, frame in enumerate(frames):
                tag = (f"ladder_t{index}" if index < targets_at
                       else f"ladder_g{index - targets_at}")
                _, reading = self._show_read(frame, tag)
                readings.append(reading)
            scores = {p["module"]: 0 for p in plans}
            for p in plans:
                m = p["module"]
                cur = tuple(f[m] for f in frames)
                if (cache is not None and moved is not None
                        and m not in moved
                        and cur == cache.get("seqs", {}).get(m)):
                    scores[m] = cache["scores"].get(m, 0)
                    continue
                for index in range(targets_at):
                    if index >= len(p["targets"]):
                        continue
                    entry = readings[index].modules[m]
                    if self._trusted(entry) \
                            and entry.char == p["targets"][index]:
                        scores[m] += 1 if p["char_index"] >= 0 else 2
                        if (p["char_index"] < 0
                                and entry.condition in ("clean", "blank")):
                            scores[m] += 1
                if p["char_index"] < 0:
                    for gi in range(guards_at):
                        if gi >= len(p["guards"]):
                            continue
                        entry = readings[targets_at + gi].modules[m]
                        if (not self._trusted(entry)
                                or entry.char != p["guards"][gi]
                                or entry.condition not in ("clean", "blank")):
                            scores[m] -= 6
            if cache is not None and moved is not None:
                cache["seqs"] = {p["module"]: tuple(f[p["module"]]
                                                    for f in frames)
                                 for p in plans}
                cache["scores"] = dict(scores)
            return scores

        self.event("phase", f"parallel cell ladder ({len(plans)} cells, "
                            f"{steps_at} steps)")
        # Baseline score BEFORE any nudge: a candidate is kept only when it
        # beats where the cell started, so an equally-wrong landing can
        # never persist a no-op offset.
        baseline = evaluate()
        for p in plans:
            p["best_score"] = baseline.get(p["module"], 0)
        cache: dict = {"scores": dict(baseline)}
        for index in range(steps_at):
            # A remaining round can cost one nudge per plan that still has
            # a candidate (the apply) plus one per loser (the revert), and
            # shows `frames_per_eval` frames. When the budgets cannot pay
            # for that, stop probing and fall through to the commit loop:
            # running the round anyway let _batch_nudge's budget guard
            # abort the whole run with every already-verified winner still
            # uncommitted.
            active = [p for p in plans
                      if not p.get("settled") and index < len(p["steps"])]
            if active and (
                    self.previews + 2 * len(active) > self.max_previews
                    or self.frames_used + frames_per_eval > self.max_frames):
                self.event("error",
                           f"cell ladder stopped before step {index + 1}: "
                           f"budget nearly spent ({self.previews}/"
                           f"{self.max_previews} previews, {self.frames_used}/"
                           f"{self.max_frames} frames); committing the "
                           f"offsets verified so far")
                for p in active:
                    # Tell the caller this cell was cut short rather than
                    # probed to exhaustion: that is a budget truncation,
                    # not a hardware verdict.
                    p["stopped_early"] = True
                break
            apply = []
            moved: set[int] = set()
            for p in plans:
                step = p["steps"][index] if index < len(p["steps"]) else None
                if step is None or p.get("settled"):
                    # Exhausted list, or the cell already reads perfect:
                    # hold at the best offset found.
                    candidate = p["best"]
                else:
                    candidate = step if p.get("absolute") \
                        else p["best"] + step
                    if p.get("cap_abs"):
                        if abs(p["cap_base"] + candidate) > p["cap"]:
                            candidate = p["best"]
                    elif abs(candidate) >= p["cap"]:
                        candidate = p["best"]  # stay inside the safe window
                p["candidate"] = candidate
                diff = candidate - p["state"]
                p["state"] = candidate
                if diff:
                    moved.add(p["module"])
                    # Record the probes actually applied so an escalated,
                    # unresolved cell can report what was tried.
                    p.setdefault("attempted", []).append(candidate)
                    apply.append((p["group"], p["local"], p["char_index"],
                                  diff))
            self._batch_nudge(apply)
            scores = evaluate(moved, cache)
            revert = []
            for p in plans:
                if scores[p["module"]] > p["best_score"]:
                    p["best_score"] = scores[p["module"]]
                    p["best"] = p["candidate"]
                    p["state"] = p["candidate"]
                else:
                    diff = p["best"] - p["state"]
                    p["state"] = p["best"]
                    if diff:
                        revert.append((p["group"], p["local"],
                                       p["char_index"], diff))
                # A perfect score cannot be beaten (every target reads
                # correctly; module plans additionally require clean flaps
                # with the guards intact): stop probing this cell so its
                # remaining candidates cost nothing - no nudges, no
                # re-reads (the cached score holds).
                perfect = (len(p["targets"])
                           if p["char_index"] >= 0
                           else 3 * len(p["targets"]))
                if p["best_score"] >= perfect:
                    p["settled"] = True
            self._batch_nudge(revert)

        improved = set()
        for p in plans:
            diff = p["best"] - p["state"]
            if diff:
                self._batch_nudge([(p["group"], p["local"],
                                    p["char_index"], diff)])
                p["state"] = p["best"]
            if p["best"] == 0 or p["best_score"] <= 0:
                continue
            key = self._ensure_cell(p["group"], p["local"], p["char_index"])
            kind = self._kind(p["char_index"])
            if p["group"] == 1:
                self._commit_local(1, key, kind, p["char_index"])
            elif self.mode == "full":
                # Remote previews are RAM-only: persist the tracked
                # absolute (overlay + residue), i.e. exactly what the
                # device holds after the ladder (committed base + this
                # ladder's best). INVARIANT: the persisted value equals the
                # value the device actually holds, so a later commit on the
                # same cell builds on this one instead of dropping it.
                target = self.live(key)
                self._guard_budgets()
                self.display.persist(p["group"], kind, target,
                                     p["local"], max(p["char_index"], 0))
                self.persists += 1
                self._wait_settled(self.timeout_s)
                self.overlay[key] = target
                self.residue[key] = 0
            if p["char_index"] < 0:
                self.module_fixed.add(p["module"])
            self._record_delta(p["module"], p["char_index"], p["targets"][0],
                               None, None, fixed=True, delta=p["best"])
            improved.add(p["module"])
            self.event("tune", f"cell ladder {p['label']}: "
                               f"{p['best']:+d} steps "
                               f"(score {p['best_score']})")
        return improved

    # -- shift-mode sweep (P1) ------------------------------------------------
    def _sweep(self) -> tuple[dict[int, dict[str, ModuleReading]], list[str]]:
        """Reverse uniform sweep over the whole drum.

        Every frame commands the SAME character on every module, so a module
        that disagrees is either misaligned or misread. The drum is walked
        backwards, so each step is ~a full revolution: every frame passes
        the magnet and re-homes, making each sample independent.
        Every drum character is shown. Returns {module: {char:
        ModuleReading}} plus the char order.
        """
        # One sweep pass consumes one sweep of the budget (MAX_SWEEPS):
        # check before starting, charge once the pass has run. Without the
        # counter the "sweep budget exhausted" guard can never fire.
        self._guard_budgets()
        chars = list(reversed(self.drum))
        readings: dict[int, dict[str, ModuleReading]] = \
            {m: {} for m in range(self.total)}
        for ch in chars:
            _, reading = self._show_read(ch * self.total, f"sw_{ord(ch)}")
            for m, entry in enumerate(reading.modules):
                readings[m][ch] = entry
        self.sweeps += 1
        return readings, chars

    def _reread_chars(self, chars, readings, tag: str):
        for ch in chars:
            _, reading = self._show_read(ch * self.total, f"{tag}_{ord(ch)}")
            for m, entry in enumerate(reading.modules):
                readings[m][ch] = entry

    def _shift(self, entry, ch: str) -> int | None:
        """Signed shift (characters) between the commanded and read glyph.

        A blank flap read against a NON-blank command is a failed sampling
        frame, not evidence of a drum shift: run-005's ':' frame read blank
        fleet-wide and poisoned every module's histogram with a phantom
        ~+10 residual, dragging purities below the trust gate. Return None
        so such samples are excluded from the histogram and denominator.
        """
        if (entry is None or not self._trusted(entry)
                or entry.char not in self.drum or ch not in self.drum
                or confusable(entry.char, ch)):
            return None
        if entry.char == " " and ch != " ":
            return None
        return -self._drum_delta(entry.char, ch)

    @staticmethod
    def _mode_purity(values) -> tuple[int | None, float, int]:
        counts = Counter(v for v in values if v is not None)
        if not counts:
            return None, 0.0, 0
        mode, count = counts.most_common(1)[0]
        return mode, count / sum(counts.values()), sum(counts.values())

    def _p1_coarse(self):
        """Reverse uniform sweep -> one-shot module offsets."""
        self.event("phase", f"P1 coarse: reverse uniform sweep "
                            f"({len(self.drum)} characters)")
        readings, chars = self._sweep()
        shifts = {m: {ch: self._shift(readings[m].get(ch), ch)
                      for ch in chars} for m in range(self.total)}
        decided: dict[int, tuple[int, int]] = {}  # module -> (steps, band)
        pre_total: dict[int, int] = {}  # module -> sum |shift| pre-move
        for m in range(self.total):
            values = [shifts[m][ch] for ch in chars]
            decisive = [v for v in values if v is not None]
            if len(decisive) < SWEEP_MIN_SAMPLES:
                self._escalate(m, "?",
                               f"unreliable reads ({len(decisive)} samples)")
                continue
            top = Counter(decisive).most_common(1)[0][1]
            if top / len(decisive) < P1_COHERENCE_MIN:
                self._escalate(
                    m, "?",
                    f"unreliable reads (no agreement: top {top}/"
                    f"{len(decisive)})")
                continue
            steps, band = p1_module_steps(values, self.steps_per_char)
            decided[m] = (steps, band)
            pre_total[m] = sum(abs(v) for v in decisive)
            npos = sum(1 for v in values if v is not None and v > 0)
            nneg = sum(1 for v in values if v is not None and v < 0)
            self.event("read", f"m{m}: P1 band {band} ({steps:+d} steps; "
                               f"+{npos}/-{nneg} of {len(chars)})")
        # Proportional module offsets (bands 1-2); band 3 moves nothing.
        for m in decided:
            self._ensure_cell(self._group_of(m), self._local_index(m), -1)
        whole = {m: steps for m, (steps, _band) in decided.items() if steps}
        if whole:
            self.event("phase", f"P1 module fixes on "
                                f"{len(whole)} modules")
            self._batch_nudge([(self._group_of(m), self._local_index(m), -1, d)
                               for m, d in sorted(whole.items())])
        # Re-read every gapped frame on a judged module — plus every frame
        # on a MOVED module, whose previously-correct characters the move
        # itself may have perturbed (stale zeros would otherwise hide the
        # damage from the residual map).
        deviant = sorted({ch for m in decided for ch in chars
                          if shifts[m][ch] not in (None, 0)
                          or decided[m][0] != 0},
                         key=self.drum.index)
        if deviant:
            self._reread_chars(deviant, readings, "p1r")
        # Sign guard: a correct move shrinks the module's total gap. If the
        # total grew, the shift->nudge polarity is inverted (or the move
        # overshot badly) — halt before persisting anything. Moved modules
        # re-read every character, so all readings below are post-move.
        for m, delta in whole.items():
            post_total = sum(
                abs(s) for ch in chars
                if (s := self._shift(readings[m].get(ch), ch)) is not None)
            if post_total > pre_total[m]:
                raise CalibError(
                    f"P1 sign guard m{m}: total gap went "
                    f"{pre_total[m]} -> {post_total} "
                    f"after {delta:+d} steps; refusing to persist")
        deviant_set = set(deviant)
        residual: dict[int, dict[str, int | None]] = {}
        for m in decided:
            out: dict[str, int | None] = {}
            for ch in chars:
                if ch in deviant_set:
                    out[ch] = self._shift(readings[m].get(ch), ch)
                else:
                    out[ch] = shifts[m][ch]
            residual[m] = out
        # Commit module fixes: local previews sit in the residue,
        # remote previews are RAM-only on the group, so persist the absolute
        # value from the recorded base there.
        for m, delta in whole.items():
            if not delta:
                continue
            group, local = self._group_of(m), self._local_index(m)
            key = self._ensure_cell(group, local, -1)
            if group == 1:
                if self.residue.get(key, 0) == 0:
                    continue
                self._commit_local(1, key, "module", -1)
            elif self.mode == "full":
                base = self._read_cell(group, local, -1)
                self._guard_budgets()
                self.display.persist(group, "module", base + delta, local, 0)
                self.persists += 1
                self._wait_settled(self.timeout_s)
                self.overlay[key] = base + delta
                self.residue[key] = 0
            self._record_delta(m, -1, "?", None, None, fixed=True,
                               delta=delta)
        # Cells still wrong or unreadable are fine-phase / hardware work.
        self._p1_flagged = sorted(
            {ch for m in decided for ch in chars
             if residual[m].get(ch) not in (None, 0)},
            key=self.drum.index)

    def _derive_flagged_readonly(self):
        """Derive the P2 flagged-char list without touching offsets.

        Used when P2 is requested without P1: a read-only reverse sweep
        plus one deviant re-read pass, mirroring the residual computation
        in `_p1_coarse` but skipping every nudge/commit. Unreliable
        modules escalate exactly as in P1.
        """
        self.event("phase", "P2 input: read-only sweep "
                            f"({len(self.drum)} characters)")
        readings, chars = self._sweep()
        shifts = {m: {ch: self._shift(readings[m].get(ch), ch)
                      for ch in chars} for m in range(self.total)}
        modes: dict[int, int] = {}
        for m in range(self.total):
            mode, purity, total = self._mode_purity(list(shifts[m].values()))
            if total < SWEEP_MIN_SAMPLES or purity < SWEEP_TRUST_PURITY:
                self._escalate(m, "?",
                               f"unreliable reads ({total} samples, "
                               f"purity {purity:.0%})")
                continue
            modes[m] = mode
        deviant = sorted({ch for m in modes for ch in chars
                          if shifts[m][ch] is not None
                          and shifts[m][ch] != modes[m]}, key=self.drum.index)
        if deviant:
            self._reread_chars(deviant, readings, "p1r")
        deviant_set = set(deviant)
        residual: dict[int, dict[str, int | None]] = {}
        for m in modes:
            out: dict[str, int | None] = {}
            for ch in chars:
                if ch in deviant_set:
                    out[ch] = self._shift(readings[m].get(ch), ch)
                else:
                    out[ch] = 0 if shifts[m][ch] == modes[m] \
                        else shifts[m][ch]
            residual[m] = out
        self._p1_flagged = sorted(
            {ch for m in modes for ch in chars
             if residual[m].get(ch) not in (None, 0)},
            key=self.drum.index)
        self.event("read", f"P2 input: {len(self._p1_flagged)} flagged "
                           f"character(s) from read-only sweep")

    def _subset_verdict(self) -> tuple[bool, str]:
        """Verdict for runs that skip acceptance.

        `converged` is only meaningful after the full acceptance sweep;
        a subset run reports what it did and leaves human judgement for
        the rest. Escalations still force `needs-human`.
        """
        if self.identity_persistent:
            mods = sorted({e["module"] for e in self.identity_persistent})
            return False, f"persistent identity/read problems on modules {mods}"
        ran = [p for p in ("p1", "p2", "p4") if p in self.phases]
        if not ran:
            return False, "no tuning or verify phases selected"
        return False, (f"subset complete ({'+'.join(ran)}); "
                       "acceptance skipped, needs-human review")

    def _p2_fine(self):
        """P2: per-character offsets from the P1 residual map."""
        flagged = list(getattr(self, "_p1_flagged", []))
        if not flagged:
            return
        self.event("phase", f"P2 fine: per-character offsets "
                            f"({len(flagged)} characters)")
        readings: dict[int, dict[str, ModuleReading]] = \
            {m: {} for m in range(self.total)}
        for ch in flagged:
            _, reading = self._show_read(ch * self.total, f"fine_{ord(ch)}")
            for m, entry in enumerate(reading.modules):
                readings[m][ch] = entry
            if all(e.source == "error" for e in reading.modules):
                # The reader could not answer this frame at all (hard
                # provider/camera failure). Retry it once: without this a
                # single transient failure escalated every module of the
                # frame and could trip the unreliable-reads abort.
                _, retry = self._show_read(ch * self.total,
                                           f"fine_{ord(ch)}r")
                for m, entry in enumerate(retry.modules):
                    readings[m][ch] = entry
        plans = []
        unreadable: set[int] = set()
        for m in range(self.total):
            for ch in flagged:
                entry = readings[m].get(ch)
                if entry is None or not self._trusted(entry):
                    # One escalation per module for the whole pass: an
                    # unreadable frame is one reader failure, not one per
                    # character.
                    if m not in unreadable:
                        unreadable.add(m)
                        self._escalate(m, ch, "unreadable during fine pass")
                    continue
                if confusable(entry.char, ch):
                    continue
                if entry.char == ch:
                    continue
                ci = self.drum.index(ch)
                # Every char-cell fault - a wrong glyph included - is walked
                # up the incremental ladder: candidate offsets in steps of
                # P2_LADDER_STEP up to the firmware's ±32 char-cell clamp. A
                # one-shot exact identity fix is never sent: the cell only
                # ever moves by a candidate within the clamp, and a fault
                # that needs a whole flap is escalated below instead of
                # written clamped.
                steps = _p2_ladder_steps(self.steps_per_char,
                                         cap=CHAR_OFFSET_LIMIT)
                plans.append({
                    "module": m,
                    "group": self._group_of(m),
                    "local": self._local_index(m),
                    "char_index": ci,
                    "steps": steps,
                    "absolute": True,
                    "cap": CHAR_OFFSET_LIMIT,
                    "targets": [ch],
                    # Character plans are scored on glyph identity alone:
                    # no condition bonus, no guard frames.
                    "guards": [],
                    "state": 0,
                    "best": 0,
                    "best_score": 0,
                    "label": f"g{self._group_of(m)} "
                             f"m{self._local_index(m)} c{ci} ({ch!r})",
                })
        if not plans:
            return
        # A ladder frame commands exactly ONE character per module, so two
        # char cells on the same module cannot be exercised in the same
        # frames: tune them in waves (one cell per module per wave) while
        # cells on different modules stay parallel.
        by_module: dict[int, list[dict]] = {}
        for p in plans:
            by_module.setdefault(p["module"], []).append(p)
        waves = max(len(cells) for cells in by_module.values())
        for index in range(waves):
            self._cell_ladder([cells[index] for cells in by_module.values()
                               if index < len(cells)])
        # A cell the ladder could not land on the commanded glyph is
        # hardware/reader work: escalate and report the probes it tried,
        # never leave a silently unresolved offset.
        for p in plans:
            if p["best_score"] >= len(p["targets"]):
                continue
            if p.get("stopped_early"):
                # The ladder ran out of budget: report the truncation, not
                # a hardware verdict (this used to claim "no working
                # offset ... tried 0 candidate(s)").
                self.event("error",
                           f"m{p['module']} c{p['char_index']}: ladder "
                           f"stopped on budget before resolving this "
                           f"cell; left unresolved")
                continue
            attempted = p.get("attempted") or []
            if not attempted:
                # Probed to exhaustion without a single applicable
                # candidate: every offset in the scan would leave the cell
                # outside the firmware's clamp (or the base is already
                # unreachable), so this IS a hardware verdict.
                self._escalate(
                    p["module"], p["targets"][0],
                    f"incremental ladder found no working offset on the "
                    f"char cell (no candidate could be applied from base "
                    f"{p.get('cap_base', 0)}; best {p['best']:+d}, "
                    f"score {p['best_score']})")
                continue
            reach = max((abs(a) for a in attempted), default=0)
            self._escalate(
                p["module"], p["targets"][0],
                f"incremental ladder found no working offset on the char "
                f"cell (tried {len(attempted)} candidate(s) up to ±{reach}; "
                f"best {p['best']:+d}, score {p['best_score']})")

    def _p4_verify(self):
        """P4: repeatability plus folded border check (short forward hops)."""
        self.event("phase", "P4 repeatability + boundary check")
        reps = [self.drum[0], self.drum[len(self.drum) // 2],
                self.drum[-1]]
        for ch in reps:
            _, first = self._show_read(ch * self.total, f"rep_{ord(ch)}")
            _, again = self._reread(ch * self.total, f"rep_{ord(ch)}r")
            for m, (a, b) in enumerate(zip(first.modules, again.modules)):
                if a.char != b.char or a.condition != b.condition:
                    self._escalate(m, ch, "unstable across repeats "
                                          f"({a.char!r}/{a.condition} vs "
                                          f"{b.char!r}/{b.condition})")
        # Short FORWARD hops across a sample of drum boundaries; the reverse
        # sweep only exercises near-full revolutions, not small hops.
        n = len(self.drum)
        for k in range(0, n - 1, 7):
            ch = self.drum[k]
            _, reading = self._show_read(ch * self.total, f"border_{k}")
            for m, entry in enumerate(reading.modules):
                if ch == " " or not self._trusted(entry):
                    continue
                if confusable(entry.char, ch):
                    continue
                if entry.char != ch:
                    self._escalate(m, ch, f"boundary check: "
                                          f"{entry.char!r}/{entry.condition}")

    def _acceptance(self) -> tuple[bool, str]:
        if self.identity_persistent:
            mods = sorted({e["module"] for e in self.identity_persistent})
            return False, f"persistent identity/read problems on modules {mods}"
        self.event("phase", "acceptance: forward sweep over the whole drum")
        bad: dict[int, list] = {}
        # forward order: small consecutive hops over the whole drum
        for ch in self.drum:
            _, reading = self._show_read(ch * self.total,
                                         f"accept_{ord(ch)}")
            for m, entry in enumerate(reading.modules):
                if ch == " ":
                    if self._trusted(entry) and entry.condition not in \
                            ("blank", "clean"):
                        bad.setdefault(m, []).append((ch, entry.condition))
                    continue
                if not self._trusted(entry):
                    bad.setdefault(m, []).append((ch, "unreadable"))
                elif confusable(entry.char, ch):
                    continue  # cannot judge this pair; not a failure
                elif entry.char != ch or entry.condition != "clean":
                    bad.setdefault(m, []).append(
                        (ch, f"{entry.char!r}/{entry.condition}"))
        if bad:
            sample = {m: v[:4] for m, v in list(bad.items())[:6]}
            return False, f"acceptance failed: {sample}"
        return True, ("acceptance: every module reads every non-space "
                      "character clean")

    # -- report helpers -------------------------------------------------------
    def summarize(self, result: str) -> dict:
        persistent = sorted({e["module"] for e in self.identity_persistent})
        offset_delta: dict[int, int] = {}
        for d in self.deltas:
            if d.get("charIndex") != -1 or d.get("proposal"):
                continue
            idx = d.get("globalModule")
            if idx is None:
                scope, local = d.get("scope", 1), d.get("module", 0)
                idx = sum(self.group_widths[: max(0, scope - 1)]) + local
            if 0 <= idx < self.total and d.get("new") is not None:
                offset_delta[idx] = d["new"] - (d.get("old") or 0)
        acceptance: dict[int, dict[str, str]] = {}
        for f in self.frames:
            if not str(f.get("tag", "")).startswith("accept_"):
                continue
            for m in f.get("modules", []):
                acceptance.setdefault(m["module"], {})[f["tag"]] = \
                    m["condition"]
        return {
            "ok": result == "converged" and not persistent,
            "result": result,
            "persistent_wrong_glyph_modules": persistent,
            "modules": [{
                "module": i,
                "offset_delta": offset_delta.get(i, 0),
                "persistent_wrong_glyph": i in persistent,
                "acceptance": acceptance.get(i, {}),
            } for i in range(self.total)],
        }

    def _phase_timing(self) -> list[dict]:
        """Per-phase seconds + budget deltas derived from phase marks.

        A mark announces the phase it starts, so a slice between two
        marks is the cost of the EARLIER one: each row is labelled with
        the previous mark's name so the numbers land on the phase that
        produced them. The slice before the first mark is "startup"; the
        closing "reverted volatile previews" marks fire after their work
        and therefore end the phase before them.
        """
        out = []
        prev = {"elapsed": 0.0, "frames": 0, "vlmCalls": 0,
                "previews": 0, "persists": 0}
        for i, mark in enumerate(self._phase_marks):
            out.append({
                "phase": (self._phase_marks[i - 1]["name"] if i
                          else "startup"),
                "seconds": round(mark["elapsed"] - prev["elapsed"], 2),
                "frames": mark["frames"] - prev["frames"],
                "vlmCalls": mark["vlmCalls"] - prev["vlmCalls"],
                "previews": mark["previews"] - prev["previews"],
                "persists": mark["persists"] - prev["persists"],
            })
            prev = mark
        return out

    # -- main -----------------------------------------------------------------
    def run(self) -> dict:
        self._t0 = time.monotonic()
        self._wall_start = time.time()
        status = self.display.status()
        contract = self.display.contract()
        if contract.get("contractVersion", 0) != SUPPORTED_CONTRACT:
            raise CalibError(f"unsupported contract "
                             f"{contract.get('contractVersion')}")
        self.total = int(status["totalModules"])
        self.charset = int(status["charset"])
        self.drum = str(status["drumOrder"])
        # Fleet geometry needs the /settings snapshot too (the firmware's
        # `masterGroupModuleCounts` is the real per-group mapping when the
        # status endpoint has no `groupWidths`).
        snapshot = self.display.snapshot()
        settings = snapshot.get("settings", snapshot) if isinstance(snapshot, dict) else {}
        self.group_widths = self._widths(status, settings)
        try:
            steps_per_rot = int(settings.get(
                "stepsPerRot", status.get("stepsPerRot", 2048)))
        except (TypeError, ValueError):
            steps_per_rot = 2048
        self.steps_per_char = max(
            1, steps_per_rot // max(1, len(self.drum))) if self.drum else 1
        self._load_remote_offsets(settings)
        report: dict = {
            "contractVersion": SUPPORTED_CONTRACT,
            "exhaustive": self.exhaustive,
            "mode": self.mode,
            "phases": list(self.phases),
            "fleet": {"totalModules": self.total,
                      "groupWidths": self.group_widths,
                      "charset": self.charset, "drumOrder": self.drum,
                      "stepsPerChar": self.steps_per_char},
            "result": "needs-human", "reason": "", "deltas": [],
            # Reproducibility: tuner params + server-supplied context
            # (camera/reader/model; never secrets) + firmware identity.
            "config": {
                "dwell_ms": self.dwell_ms, "timeout_s": self.timeout_s,
                "min_confidence": self.min_confidence,
                "max_seconds": self.max_seconds,
                "stepsPerRot": steps_per_rot,
                **self.run_context,
            },
            "firmware": {
                "contractVersion": contract.get("contractVersion"),
                "schemaVersion": status.get("schemaVersion"),
                "mode": status.get("mode"),
            },
        }
        with open(os.path.join(self.photo_dir, "snapshot.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(snapshot, fh)
        try:
            self.display.hold(True)
            # Drop any uncommitted preview nudges from earlier runs before
            # reading the baseline: live offsets must equal persisted NVS.
            self._clear_volatile_previews()
            self._p0_register()
            if self.mode == "dry-run":
                readings, chars = self._sweep()
                for m in range(self.total):
                    mode, purity, total = self._mode_purity(
                        [self._shift(readings[m].get(ch), ch)
                         for ch in chars])
                    self.event("read", f"m{m}: shift mode "
                                       f"{mode if mode is not None else '?'}"
                                       f" ({purity:.0%} of {total})")
                ok, reason = False, ("dry-run: shift table only, "
                                     "nothing applied")
            else:
                if "p1" in self.phases:
                    self._p1_coarse()
                elif "p2" in self.phases:
                    # P2 needs the P1 residual map; without P1 run a
                    # read-only sweep to derive it (no nudges/commits).
                    self._derive_flagged_readonly()
                if "p2" in self.phases:
                    self._p2_fine()
                if "p4" in self.phases:
                    self._p4_verify()
                if "acceptance" in self.phases:
                    ok, reason = self._acceptance()
                else:
                    ok, reason = self._subset_verdict()
            report["result"] = "converged" if ok else "needs-human"
            report["reason"] = reason
        except CalibError as exc:
            report["reason"] = str(exc)
            report["traceback"] = traceback.format_exc(limit=8)
            if self.aborted:
                report["reason"] = "aborted by user"
            # Phase commits are verified and kept (commit-per-phase design):
            # only volatile previews from the interrupted phase are dropped.
            # The pre-run snapshot stays on disk for a manual rollback.
            self._clear_volatile_previews()
        finally:
            # Full mode leaves no uncommitted nudges: persisted winners stay,
            # ghost residue on escalated cells is dropped. Preview/dry-run
            # intentionally leave their volatile previews for inspection.
            if self.mode == "full":
                self._clear_volatile_previews()
            try:
                self.display.hold(False)
            except CalibError:
                pass
            report["frames"] = self.frames
            report["deltas"] = self.deltas
            report["identity"] = {"persistent": self.identity_persistent}
            if self.mode == "full":
                report["skipped"] = [
                    p for p in ("p1", "p2", "p4", "acceptance")
                    if p not in self.phases]
            else:
                report["skipped"] = []
            report["summary"] = self.summarize(report["result"])
            report["budgets"] = {"frames": self.frames_used,
                                 "vlmCalls": self.vlm_calls,
                                 "previews": self.previews,
                                 "persists": self.persists}
            report["timing"] = {
                "wallSeconds": round(time.monotonic() - self._t0, 1),
                "wallStart": time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime(self._wall_start)),
                "vlmSeconds": round(self._vlm_seconds, 1),
                "vlmTokens": dict(self._vlm_tokens),
                "phases": self._phase_timing(),
            }
            with open(os.path.join(self.photo_dir, "report.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
            self.report = report
            self.event("done", f"{report['result']}: {report['reason']}")
        return report
