"""VLM-reader calibration state machine (P0 -> P4 + acceptance).

Deterministic; the VLM is only a reader (see reader.py). Offset math is
exact because drums move forward-only and `drumOrder` is known: a module
that reads `seen` when commanded `target` is off by
`(drumIndex(target) - drumIndex(seen)) mod drumLen` forward steps.

Fleet semantics mirror tools/calib: group 1 tunes with volatile previews
then persists the winner; remote groups (master-only access, no preview
in firmware) use persist-verify-revert per cell.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from collections import Counter

import cv2

from calib.display import CalibError

from .reader import (ModuleReading, ReaderError, Reading, annotate_modules,
                     jpeg_bytes)

SUPPORTED_CONTRACT = 1
# Wear/time budgets.
MAX_FRAMES = 300
MAX_VLM_CALLS = 300
MAX_PREVIEWS = 200
MAX_PERSISTS = 400
MAX_SWEEPS = 3
MAX_TUNE_ITER = 3
# Abort when the reader cannot be trusted: more than this many
# unreliable/unreadable escalations means camera/framing/lighting is bad
# and continuing would burn wear/budget tuning noise.
MAX_UNRELIABLE_READS = 5
# Deltas tried per suspect cell (motor steps), coarse first.
TRY_DELTAS = (4, -4, 2, -2, 8, -8, 1, -1)
CONDITION_COST = {"clean": 0, "blank": 0, "half": 1, "double": 2,
                  "stuck": 3, "unreadable": 4}
# A module misaligned (half/double, identity still right) on this many
# uniform frames is a whole-drum phase fault: tune the module cell once
# instead of scattering per-char alignment searches across the drum.
ALIGN_VOTE_MIN = 2
# Sub-pitch module-offset trims: a module offset slides every character's
# landing by the same steps, so a fraction of one flap pulls characters
# sitting just past their flap boundary back onto their own flap without
# moving well-centred characters. Multiples of a full character stay on the
# coarse `_identity_steps` path.
MODULE_TRIM_FRACTIONS = (0.5, 0.25, 0.125, 0.0625)
# Distinct wrong glyphs evaluated per suspect module during the trim.
MODULE_TRIM_TARGETS = 3
# Correct neighbours checked so a trim that fixes the targets by breaking
# the surrounding characters is rejected.
MODULE_TRIM_GUARDS = 3
# Firmware limits (src/CalibApi.h + SplitFlapWebServer.cpp): char offset
# cells are motor steps clamped to ±32, and /api/calib/preview rejects a
# single |delta| > 32 with HTTP 400.
CHAR_OFFSET_LIMIT = 32
PREVIEW_DELTA_MAX = 32
# Shift-mode sweep (the new P1): command every character uniformly on every
# module, walking the drum in REVERSE so each step is ~a full revolution and
# every frame passes the magnet (independently homed). The per-module
# histogram of (seen - commanded) then gives the module offset in one shot.
SWEEP_MIN_SAMPLES = 24       # trusted samples needed to judge a module
SWEEP_TRUST_PURITY = 0.80    # dominant-shift share needed to trust it
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
# Sub-pitch cell trims (coarse -> fine, cumulative). Applied to module cells
# for boundary residuals and to char cells for per-character faults; all
# candidates stay below one character pitch.
# (MODULE_TRIM_FRACTIONS defined once above with the alignment constants.)


def confusable(a: str, b: str) -> bool:
    """True when two glyphs cannot be told apart on the drum."""
    return a != b and b in CONFUSABLES.get(a, ())


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


class VlmCalibrator:
    def __init__(self, display, camera, reader, photo_dir: str,
                 dwell_ms: int = 800, timeout_s: float = 60.0,
                 min_confidence: float = 0.6, exhaustive: bool = False,
                 mode: str = "full", on_event=None, max_seconds: float = 3600.0,
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
        # character (delta * steps_per_char); alignment fixes move a few
        # steps (TRY_DELTAS).
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
            # derive per-phase seconds + frames/VLM/preview/persist cost.
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
                raise CalibError("VLM call budget exhausted")
        if preview and self.previews >= self.max_previews:
            raise CalibError("preview budget exhausted")
        if self.persists >= self.max_persists:
            raise CalibError("persist budget exhausted")
        if self.sweeps >= self.max_sweeps:
            raise CalibError("sweep budget exhausted")
        if time.monotonic() - self._t0 > self.max_seconds:
            raise CalibError("time budget exhausted")

    def _charge_vlm(self, calls: int) -> None:
        """Account VLM round trips after a frame is read."""
        self.vlm_calls += calls
        if self.vlm_calls > self.max_vlm_calls:
            raise CalibError("VLM call budget exhausted")

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
        model = getattr(vlm, "model", None) or "?"
        t0 = time.monotonic()
        try:
            reading = self.reader.read(jpeg_bytes(send), total=self.total,
                                       expected=frame, charset=self.drum,
                                       drum=self.drum)
            self._reader_failures = 0
        except ReaderError as exc:
            self._reader_failures += 1
            reading = self.reader.error_reading(self.total, frame, str(exc))
            self.event("error", f"reader failed on {tag}: {exc}")
            if self._reader_failures >= 3:
                raise CalibError(f"VLM reader failing repeatedly: {exc}") from exc
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
        self.event("vlm", f"POST chat model={model} tag={tag} "
                           f"->{calls} call(s) in {dt:.1f}s{tok}: "
                           f"{self._read_line(frame, reading)}")
        # Account the real VLM round trips: a read may re-ask once, and
        # failed parses still consumed provider calls.
        self._charge_vlm(calls)
        rec = {"tag": tag, "frameId": frame_id, "frame": frame,
               "photo": os.path.basename(path)}
        rec.update(reading.as_dict())
        self.frames.append(rec)
        self.event("read", f"{tag}: {self._read_line(frame, reading)} "
                           f"({self._summary_line(reading)})",
                   rec["photo"], detail=rec)
        return rec, reading

    # -- fleet geometry (mirrors calib.loop.Calibrator) -----------------------
    def _widths(self, status: dict) -> list[int]:
        total = int(status["totalModules"])
        local = int(status["numModules"])
        groups = int(status.get("groupCount", 1)) or 1
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
        return (entry.source == "vlm"
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

    def _identity_steps(self, seen: str, target: str,
                        char_index: int = 0) -> int | None:
        """Signed-minimal motor-step correction moving `seen` to `target`.

        A per-char cell shifts that character forward for positive steps,
        but a module cell re-anchors the homing magnet reference for the
        whole drum and shifts the displayed character the other way: on
        magnet detection the firmware sets `position = magnetPos +
        moduleOffset`, then steps forward-only to `charPosition`. Raising
        the module offset therefore shows an *earlier* drum character, so
        module-cell corrections are negated. Getting this backwards makes
        the tune walk the drum away from the target instead of towards it.
        """
        chars = self._drum_delta(seen, target)
        if chars is None:
            return None
        steps = chars * self.steps_per_char
        return -steps if char_index < 0 else steps

    def _drum_error(self, seen: str, target: str) -> int | None:
        """Absolute drum distance in characters (sign-independent)."""
        chars = self._drum_delta(seen, target)
        return None if chars is None else abs(chars)

    def _settle_for_move(self, steps: int) -> None:
        """Wait for the display to settle after a correction of `steps`.

        The firmware reports idle when the command drains, but a flap can
        still be travelling (re-homes at maxVel take seconds). Scale the
        extra settle by move size on top of the normal wait so tune reads
        never photograph a moving flap.
        """
        extra = min(10.0, abs(steps) / max(1, self.steps_per_char) * 0.5)
        if extra > 0.1 and self.dwell_ms > 0:
            time.sleep(extra)
        self._wait_settled(self.timeout_s)

    def _steady_read(self, show_frame: str, tag: str, module: int,
                     target: str) -> tuple[dict, Reading]:
        """Read until two consecutive reads agree, else escalate the blur.

        A flap photographed mid-travel reads as a random wrong glyph at
        high confidence (the run-001 tune photos show motion blur read as
        '8' then 'Q'). If two back-to-back reads of the same settled frame
        disagree by more than one drum position, the flap is still moving:
        wait and re-read once; if still unstable, return None so the caller
        escalates instead of correcting a transient.
        """
        rec, reading = self._show_read(show_frame, tag)
        first = reading.modules[module]
        if not self._trusted(first) or first.char not in self.drum:
            return rec, reading
        if self.dwell_ms > 0:
            time.sleep(0.5)
        # Re-reads, not re-shows: the drum is already at the frame.
        rec2, reading2 = self._reread(show_frame, tag + "_steady")
        second = reading2.modules[module]
        if not self._trusted(second) or second.char not in self.drum:
            return rec2, reading2
        d1 = self._drum_error(first.char, target)
        d2 = self._drum_error(second.char, target)
        if d1 is None or d2 is None:
            return rec2, reading2
        if abs(d1 - d2) > 1:
            self.event("error",
                       f"m{module} reading unstable between consecutive reads "
                       f"({first.char!r} vs {second.char!r}; flap still moving?)")
            if self.dwell_ms > 0:
                time.sleep(1.0)
            rec3, reading3 = self._reread(show_frame, tag + "_steady2")
            third = reading3.modules[module]
            if not self._trusted(third) or third.char not in self.drum:
                return rec3, reading3
            d3 = self._drum_error(third.char, target)
            if d3 is None or abs(d3 - d2) > 1:
                return rec3, reading3  # caller sees non-convergence via stalls
        return rec2, reading2

    def _steady_confirm(self, show_frame: str, tag: str, module: int,
                        target: str, reading: Reading) -> Reading:
        """Blur check for a `_reread` result: one more agreeing read.

        Mirrors the tail of `_steady_read` but never shows (the drum is
        already at the frame). Returns the confirming reading, or the
        original when no confirmation is possible.
        """
        first = reading.modules[module]
        if not self._trusted(first) or first.char not in self.drum:
            return reading
        if self.dwell_ms > 0:
            time.sleep(0.5)
        _, reading2 = self._reread(show_frame, tag + "_steady")
        second = reading2.modules[module]
        if not self._trusted(second) or second.char not in self.drum:
            return reading2
        d1 = self._drum_error(first.char, target)
        d2 = self._drum_error(second.char, target)
        if d1 is not None and d2 is not None and abs(d1 - d2) > 1:
            self.event("error",
                       f"m{module} reading unstable between consecutive reads "
                       f"({first.char!r} vs {second.char!r}; flap still moving?)")
        return reading2

    def _cost(self, entry: ModuleReading, target: str) -> tuple:
        return (0 if entry.char == target else 1,
                CONDITION_COST.get(entry.condition, 4),
                -entry.confidence)

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

    def _apply_identity_delta(self, delta: int, char_index: int, group: int,
                              local: int, key, kind: str) -> bool:
        """Apply an identity correction to one cell; False when impossible.

        Char cells are motor-step offsets the firmware clamps to ±32, so a
        whole-character correction (stepsPerChar motor steps, 55 at the
        default settings) can never live there: returning False makes the
        caller escalate instead of writing a silently clamped wrong value.
        Module cells are unconstrained and re-home the whole drum, so the
        full correction is sent as ONE preview (the endpoint allows it for
        charIndex < 0): the module re-homes once instead of once per 32
        steps. Char cells stay chunked because the firmware clamps their
        absolute value to ±32.
        """
        target_value = self.live(key) + delta
        if char_index >= 0 and abs(target_value) > CHAR_OFFSET_LIMIT:
            return False
        if group == 1:
            chunks = [delta] if char_index < 0 and delta else _preview_chunks(delta)
            for chunk in chunks:
                self._guard_budgets(preview=True)
                self.display.preview(local, char_index, chunk)
                residue = self.residue.get(key, 0) + chunk
                if char_index >= 0:
                    # Mirror the firmware: each preview clamps the cell's
                    # ABSOLUTE value to ±32, so must the tracked belief.
                    absolute = self.overlay.get(key, 0) + residue
                    residue = (max(-CHAR_OFFSET_LIMIT,
                                   min(CHAR_OFFSET_LIMIT, absolute))
                               - self.overlay.get(key, 0))
                self.residue[key] = residue
                self.previews += 1
                self._wait_settled(self.timeout_s)
        else:
            self._guard_budgets()
            self.display.persist(group, kind, target_value, local,
                                 max(char_index, 0))
            self.persists += 1
            self.overlay[key] = target_value
            self.residue[key] = 0
            self._wait_settled(self.timeout_s)
        return True

    def _tune_identity(self, module: int, char_index: int, target: str,
                       show_frame: str) -> dict:
        """Drive `module` until it reads `target` (character-level).

        Signed-minimal drum deltas with move-scaled settle; steady reads
        reject motion-blur transients. Two successive non-improving
        iterations escalate instead of burning the preview budget.
        """
        group = self._group_of(module)
        local = self._local_index(module)
        key = self._ensure_cell(group, local, char_index)
        kind = self._kind(char_index)
        tag = f"tune_g{group}m{local}c{char_index}"
        before: ModuleReading | None = None
        last_err: int | None = None
        stalls = 0
        first_pass = True
        for _ in range(MAX_TUNE_ITER):
            if first_pass:
                # Show once: the drum is not at this frame yet.
                _, reading = self._steady_read(show_frame, tag, module,
                                               target)
                first_pass = False
            else:
                # Already shown: the preview re-homed in place, so settle
                # + re-read without driving another full revolution.
                _, reading = self._reread(show_frame, tag)
                entry_probe = reading.modules[module]
                if (self._trusted(entry_probe)
                        and entry_probe.char in self.drum):
                    reading = self._steady_confirm(show_frame, tag, module,
                                                   target, reading)
            entry = reading.modules[module]
            if before is None:
                before = entry
            if entry.char == target and entry.condition == "clean":
                self._commit_local(group, key, kind, char_index)
                return self._record_delta(module, char_index, target, before,
                                          entry, fixed=True, delta=0)
            if self.mode == "dry-run" or (self.mode == "preview" and group != 1):
                delta = self._identity_steps(entry.char, target, char_index)
                return self._record_delta(module, char_index, target, before,
                                          entry, fixed=False, proposal=True,
                                          delta=delta)
            if not self._trusted(entry):
                self._escalate(module, target,
                               "photo unreadable while tuning")
                return self._record_delta(module, char_index, target, before,
                                          entry, fixed=False)
            if entry.char == target:
                # Identity is right but the flap is misaligned: fine-tune.
                return self._tune_alignment(module, char_index, target,
                                            show_frame, before=before)
            err = self._drum_error(entry.char, target)
            if err is None:
                self._escalate(module, target,
                               f"read {entry.char!r} is not on the drum")
                return self._record_delta(module, char_index, target, before,
                                          entry, fixed=False)
            if last_err is not None:
                if err < last_err:
                    stalls = 0
                else:
                    stalls += 1
                    if stalls >= 2:
                        self._escalate(
                            module, target,
                            f"correction not converging (drum error {last_err} "
                            f"-> {err} chars); escalating instead of burning "
                            f"more previews")
                        return self._record_delta(module, char_index, target,
                                                  before, entry, fixed=False)
            last_err = err
            delta = self._identity_steps(entry.char, target, char_index)
            if not delta:
                self._escalate(module, target,
                               f"read {entry.char!r} is not on the drum")
                return self._record_delta(module, char_index, target, before,
                                          entry, fixed=False)
            if not self._apply_identity_delta(delta, char_index, group, local,
                                              key, kind):
                self._escalate(
                    module, target,
                    f"identity fix {delta:+d} motor steps does not fit a "
                    f"char cell (firmware clamps char offsets to "
                    f"±{CHAR_OFFSET_LIMIT})")
                return self._record_delta(module, char_index, target, before,
                                          entry, fixed=False)
            self._settle_for_move(delta)
        _, reading = self._show_read(show_frame, tag + "_final")
        after = reading.modules[module]
        fixed = after.char == target and after.condition == "clean"
        if fixed:
            self._commit_local(group, key, kind, char_index)
        return self._record_delta(module, char_index, target, before, after,
                                  fixed=fixed)

    def _tune_alignment(self, module: int, char_index: int, target: str,
                        show_frame: str,
                        before: ModuleReading | None = None) -> dict:
        """Small-step search for a clean flap while keeping identity right."""
        group = self._group_of(module)
        local = self._local_index(module)
        key = self._ensure_cell(group, local, char_index)
        kind = self._kind(char_index)
        tag = f"align_g{group}m{local}c{char_index}"
        if before is None:
            _, reading = self._show_read(show_frame, tag)
            before = reading.modules[module]
        best = self._cost(before, target)
        if best[0] == 0 and best[1] == 0:
            return self._record_delta(module, char_index, target, before,
                                      before, fixed=True)
        if self.mode == "dry-run" or (self.mode == "preview" and group != 1):
            return self._record_delta(module, char_index, target, before,
                                      before, fixed=False, proposal=True)
        if group != 1:
            # Remote: persist absolute candidates, verify, revert losers.
            base = self.live(key)
            best_abs = base
            for d in TRY_DELTAS:
                if char_index >= 0 and abs(base + d) > CHAR_OFFSET_LIMIT:
                    continue  # firmware clamps char cells; not a real candidate
                self._guard_budgets()
                self.display.persist(group, kind, base + d, local,
                                     max(char_index, 0))
                self.persists += 1
                self._wait_settled(self.timeout_s)
                _, reading = self._show_read(show_frame, tag)
                entry = reading.modules[module]
                cost = self._cost(entry, target)
                if cost < best:
                    best, best_abs = cost, base + d
                else:
                    self.display.persist(group, kind, best_abs, local,
                                         max(char_index, 0))
                    self.persists += 1
                    self._wait_settled(self.timeout_s)
            self.overlay[key] = best_abs
            self.residue[key] = 0
        else:
            # Local: volatile preview ladder. Each candidate is applied
            # FROM THE BASE (not compounded), because the deltas are not a
            # fresh search from wherever the last candidate left the flap.
            base_live = self.live(key)
            applied = 0
            best_extra = 0
            for d in TRY_DELTAS:
                if char_index >= 0 and abs(base_live + d) > CHAR_OFFSET_LIMIT:
                    continue  # firmware clamps char cells; not a real candidate
                correction = d - applied
                self._guard_budgets(preview=True)
                self.display.preview(local, char_index, correction)
                self.residue[key] = self.residue.get(key, 0) + correction
                applied = d
                self.previews += 1
                self._wait_settled(self.timeout_s)
                _, reading = self._show_read(show_frame, tag)
                cost = self._cost(reading.modules[module], target)
                if cost < best:
                    best = cost
                    best_extra = d
            if self.mode == "full":
                self._guard_budgets()
                self.display.persist(group, kind, base_live + best_extra, local,
                                     max(char_index, 0))
                self.persists += 1
                self.overlay[key] = base_live + best_extra
                self.residue[key] = 0
                self._wait_settled(self.timeout_s)
            else:
                correction = (base_live + best_extra) - self.live(key)
                if correction:
                    self.display.preview(local, char_index, correction)
                    self.residue[key] = self.residue.get(key, 0) + correction
                    self.previews += 1
                    self._wait_settled(self.timeout_s)
        _, reading = self._show_read(show_frame, tag + "_verify")
        after = reading.modules[module]
        fixed = after.char == target and after.condition == "clean"
        return self._record_delta(module, char_index, target, before, after,
                                  fixed=fixed)

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
        """Apply cell nudges [(group, local, char_index, delta), ...] in one pass.

        Group 1 is applied locally; groups 2..6 are forwarded by the master
        over ESP-NOW and applied RAM-only on each remote (the master's busy
        fence covers their homing via preview acks). N cells across the
        fleet cost one homing pass, not N. Falls back to serial single
        previews on older firmware (local group only).
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
                batch([{"scope": group, "module": local,
                        "charIndex": char_index, "delta": delta}
                       for local, char_index, delta in items])
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

    def _module_trim(self, votes: dict[int, list[str]],
                     seen: dict[int, dict[str, str]],
                     glyphs: list[str],
                     offset_base: dict[int, int] | None = None) -> set[int]:
        """Sub-pitch module-offset trim, run in parallel across modules.

        A module offset shifts every landing by a fraction of a character,
        so a trim smaller than one flap pulls boundary characters (showing
        the next flap) back without moving the characters that are centred.
        Every suspect module applies its own candidate in the SAME mixed
        frames/reads, and the batch preview homes them in one pass, so the
        cost is `rounds x frames`, not `modules x rounds x frames`.

        Modules wrong on every tested glyph (a whole-drum shift) or on
        glyphs pointing both ways (no single shift helps) are left to the
        coarse/per-character paths. Returns the modules that improved.

        Mode gating (same contract as _tune_identity/_tune_alignment):
        dry-run touches nothing (the surviving votes become proposals in
        the later paths); preview may nudge the LOCAL group only, since
        remote trims can only be committed via persist; full does both.
        """
        if self.mode == "dry-run":
            return set()
        plans = []
        wrong_sets = {m: set(g) for m, g in votes.items()}
        for module in sorted(votes):
            wrong = [g for g in glyphs if g in wrong_sets[module]]
            if not wrong or len(wrong) >= len(glyphs):
                continue
            if self.mode == "preview" and self._group_of(module) != 1:
                continue  # remote trim needs persist; not allowed in preview
            errors = []
            for glyph in wrong:
                delta = self._drum_delta(seen[module].get(glyph, "?"), glyph)
                if delta:
                    errors.append(delta)
            if not errors:
                continue
            if len({-1 if e < 0 else 1 for e in errors}) != 1:
                continue  # mixed direction: no single offset can help
            wrong_set = set(wrong)
            right = [g for g in glyphs if g not in wrong_set]
            # Guards: correct glyphs spread across the drum. A trim shifts
            # every other character too, so any candidate that breaks the
            # surrounding characters is rejected before it is committed.
            if right:
                stride = max(1, len(right) // (MODULE_TRIM_GUARDS + 1))
                guards = right[::stride][:MODULE_TRIM_GUARDS]
            else:
                fallback = "E" if "E" in self.drum else self.drum[1]
                if fallback in wrong_set:
                    fallback = self.drum[1]
                guards = [fallback]
            local = self._local_index(module)
            magnitudes = [max(1, int(round(self.steps_per_char * f)))
                          for f in MODULE_TRIM_FRACTIONS]
            direction = -1 if errors[0] > 0 else 1
            plans.append({
                "module": module,
                "group": self._group_of(module),
                "local": local,
                "char_index": -1,
                "direction": direction,
                "steps": [direction * m for m in magnitudes],
                "cap": self.steps_per_char,
                "offset_base": (offset_base or {}).get(module, 0),
                "targets": wrong[:MODULE_TRIM_TARGETS],
                "guards": guards[:MODULE_TRIM_GUARDS],
                "state": 0,
                "best": 0,
                "best_score": 0,
                "label": f"g{self._group_of(module)} m{local} c-1",
            })
        if not plans:
            return set()
        return self._cell_ladder(plans)

    def _cell_ladder(self, plans: list[dict]) -> set[int]:
        """Parallel coarse-to-fine ladder over independent cell offsets.

        Plans are cells (module offsets or per-character cells). Every plan
        applies its own candidate in the SAME mixed frames/reads and the
        batch preview homes the touched modules in one pass, so the cost is
        `steps x frames`, not `plans x steps x frames`. A candidate is kept
        only when it raises the plan's score (targets correct+clean minus
        guard breakage), so the smallest clearing shift wins.
        """
        if not plans or self.mode == "dry-run":
            return set()
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

        def frame_for(index: int) -> str:
            cells = []
            for module in range(self.total):
                p = by_module.get(module)
                if p is None:
                    cells.append("E")
                elif index < len(p["targets"]):
                    cells.append(p["targets"][index])
                elif index - targets_at < len(p["guards"]):
                    cells.append(p["guards"][index - targets_at])
                else:
                    cells.append(p["guards"][0])
            return "".join(cells)

        def evaluate() -> dict[int, int]:
            scores = {p["module"]: 0 for p in plans}
            for index in range(targets_at):
                _, reading = self._show_read(frame_for(index),
                                             f"ladder_t{index}")
                for p in plans:
                    if index >= len(p["targets"]):
                        continue
                    entry = reading.modules[p["module"]]
                    if self._trusted(entry) and entry.char == p["targets"][index]:
                        scores[p["module"]] += 2
                        if entry.condition in ("clean", "blank"):
                            scores[p["module"]] += 1
            for index in range(guards_at):
                _, reading = self._show_read(frame_for(targets_at + index),
                                             f"ladder_g{index}")
                for p in plans:
                    if index >= len(p["guards"]):
                        continue
                    entry = reading.modules[p["module"]]
                    if (not self._trusted(entry)
                            or entry.char != p["guards"][index]
                            or entry.condition not in ("clean", "blank")):
                        scores[p["module"]] -= 6
            return scores

        self.event("phase", f"parallel cell ladder ({len(plans)} cells, "
                            f"{steps_at} steps)")
        # Baseline score BEFORE any nudge: a candidate is kept only when it
        # beats where the cell started. Without this a correct-char/half-flap
        # target (score 2) would accept the first candidate that leaves it
        # equally half (score 2 > 0) and persist a no-op offset.
        baseline = evaluate()
        for p in plans:
            p["best_score"] = baseline.get(p["module"], 0)
        for index in range(steps_at):
            apply = []
            for p in plans:
                step = p["steps"][index] if index < len(p["steps"]) else 0
                candidate = step if p.get("absolute") else p["best"] + step
                if p.get("cap_abs"):
                    if abs(p["cap_base"] + candidate) > p["cap"]:
                        candidate = p["best"]
                elif abs(candidate) >= p["cap"]:
                    candidate = p["best"]  # stay inside the safe window
                p["candidate"] = candidate
                diff = candidate - p["state"]
                p["state"] = candidate
                if diff:
                    apply.append((p["group"], p["local"], p["char_index"],
                                  diff))
            self._batch_nudge(apply)
            scores = evaluate()
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
                # Remote previews are RAM-only: persist the absolute winner
                # (base + whole-phase offset already previewed + ladder) so
                # the fix survives the revert.
                base = self._read_cell(p["group"], p["local"],
                                       p["char_index"])
                target = base + p.get("offset_base", 0) + p["best"]
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
        Returns {module: {char: ModuleReading}} plus the char order.
        """
        chars = list(reversed(self.drum))
        readings: dict[int, dict[str, ModuleReading]] = \
            {m: {} for m in range(self.total)}
        for ch in chars:
            _, reading = self._show_read(ch * self.total, f"sw_{ord(ch)}")
            for m, entry in enumerate(reading.modules):
                readings[m][ch] = entry
        return readings, chars

    def _reread_chars(self, chars, readings, tag: str):
        for ch in chars:
            _, reading = self._show_read(ch * self.total, f"{tag}_{ord(ch)}")
            for m, entry in enumerate(reading.modules):
                readings[m][ch] = entry

    def _shift(self, entry, ch: str) -> int | None:
        """Signed shift (characters) between the commanded and read glyph."""
        if (entry is None or not self._trusted(entry)
                or entry.char not in self.drum or ch not in self.drum
                or confusable(entry.char, ch)):
            return None
        return -self._drum_delta(entry.char, ch)

    @staticmethod
    def _mode_purity(values) -> tuple[int | None, float, int]:
        counts = Counter(v for v in values if v is not None)
        if not counts:
            return None, 0.0, 0
        mode, count = counts.most_common(1)[0]
        return mode, count / sum(counts.values()), sum(counts.values())

    def _module_phase_trim(self, phase_cells: dict[int, list[str]],
                           readings, chars: list[str],
                           offset_base: dict[int, int] | None = None) -> set[int]:
        """Fine module-cell search for a whole-drum seam (phase) fault.

        Identity is right but several characters show half/double: a small
        module-cell nudge centres the seam. The reader reports no magnitude,
        so scan the classic fine ladder around the base (both directions).
        """
        plans = []
        magnitudes = TRY_DELTAS
        for m, cells in sorted(phase_cells.items()):
            if self.mode == "preview" and self._group_of(m) != 1:
                continue
            right = [ch for ch in chars
                     if ch not in cells
                     and readings[m].get(ch) is not None
                     and self._trusted(readings[m][ch])
                     and readings[m][ch].char == ch]
            if right:
                stride = max(1, len(right) // (MODULE_TRIM_GUARDS + 1))
                guards = right[::stride][:MODULE_TRIM_GUARDS]
            else:
                guards = ["E"]
            plans.append({
                "module": m,
                "group": self._group_of(m),
                "local": self._local_index(m),
                "char_index": -1,
                "steps": list(magnitudes),
                "absolute": True,
                "cap": self.steps_per_char,
                "offset_base": (offset_base or {}).get(m, 0),
                "targets": cells[:MODULE_TRIM_TARGETS],
                "guards": guards,
                "state": 0,
                "best": 0,
                "best_score": 0,
                "label": f"g{self._group_of(m)} m{self._local_index(m)} "
                         f"phase",
            })
        if plans:
            return self._cell_ladder(plans)
        return set()

    def _p1_coarse(self):
        """Reverse uniform sweep -> one-shot module offsets + sub-pitch trim."""
        self.event("phase", f"P1 coarse: reverse uniform sweep "
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
            self.event("read", f"m{m}: shift mode {mode:+d} "
                               f"({purity:.0%} of {total})")
        # One-shot whole-character offsets.
        for m in modes:
            self._ensure_cell(self._group_of(m), self._local_index(m), -1)
        whole = {m: mode * self.steps_per_char
                 for m, mode in modes.items() if mode}
        if whole:
            self.event("phase", f"P1 whole-character fixes on "
                                f"{len(whole)} modules")
            self._batch_nudge([(self._group_of(m), self._local_index(m), -1, d)
                               for m, d in sorted(whole.items())])
        # Re-read every frame where a trusted module deviated from its mode.
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
        # Sub-pitch module trim for a minority of same-sign +/-1 residuals.
        votes: dict[int, list[str]] = {}
        seen: dict[int, dict[str, str]] = {}
        minority = max(1, len(chars) // 2)
        for m, res in residual.items():
            outliers = [ch for ch in chars if res.get(ch) not in (None, 0)]
            if not outliers or len(outliers) > minority:
                continue
            vals = [res[ch] for ch in outliers]
            if any(abs(v) != 1 for v in vals):
                continue
            if len({v > 0 for v in vals}) != 1:
                continue
            if self.mode == "preview" and self._group_of(m) != 1:
                continue
            votes[m] = outliers
            seen[m] = {ch: readings[m][ch].char for ch in outliers}
        if votes:
            self.event("phase", f"P1 sub-pitch module trim "
                                f"({len(votes)} modules)")
            trimmed = self._module_trim(votes, seen, chars, whole)
        else:
            trimmed = set()
        # Whole-drum seam fault: several half/double cells on one module.
        phase_cells: dict[int, list[str]] = {}
        for m in modes:
            cells = []
            for ch in chars:
                entry = readings[m].get(ch)
                if entry is None or not self._trusted(entry):
                    continue
                if entry.char != ch and not confusable(entry.char, ch):
                    continue
                if entry.condition in ("half", "double"):
                    cells.append(ch)
            if cells:
                phase_cells[m] = cells
        phase_votes = {m: cells for m, cells in phase_cells.items()
                       if len(cells) >= ALIGN_VOTE_MIN}
        if phase_votes:
            self.event("phase", f"P1 module phase trim "
                                f"({len(phase_votes)} modules)")
            phased = self._module_phase_trim(phase_votes, readings, chars,
                                             whole)
        else:
            phased = set()
        handled = trimmed | phased
        # Commit whole-character fixes: local previews sit in the residue,
        # remote previews are RAM-only on the group, so persist the absolute
        # value from the recorded base there.
        for m, delta in whole.items():
            if not delta or m in handled:
                # Trimmed/phased modules already committed base + delta +
                # ladder in one absolute write.
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
            {ch for m in modes for ch in chars
             if residual[m].get(ch) not in (None, 0)}
            | {ch for cells in phase_cells.values() for ch in cells},
            key=self.drum.index)

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
        plans = []
        for m in range(self.total):
            for ch in flagged:
                entry = readings[m].get(ch)
                if entry is None or not self._trusted(entry):
                    self._escalate(m, ch, "unreadable during fine pass")
                    continue
                if confusable(entry.char, ch):
                    continue
                if entry.char == ch and entry.condition == "clean":
                    continue
                ci = self.drum.index(ch)
                right = [g for g in self.drum
                         if g != " " and g != ch and g != entry.char]
                stride = max(1, len(right) // (MODULE_TRIM_GUARDS + 1))
                guards = right[::stride][:MODULE_TRIM_GUARDS]
                if entry.char != ch:
                    full = self._identity_steps(entry.char, ch, ci)
                    if full is None:
                        self._escalate(m, ch, f"read {entry.char!r} is not "
                                              f"on the drum")
                        continue
                    key = self._ensure_cell(self._group_of(m),
                                            self._local_index(m), ci)
                    if abs(self.live(key) + full) > CHAR_OFFSET_LIMIT:
                        self._escalate(m, ch, "identity fix does not fit a "
                                              "char cell (firmware clamps "
                                              "char offsets to ±32)")
                        continue
                    # One-shot: the exact landing correction, then verify.
                    steps = [full]
                else:
                    # Right glyph, seam off-phase: the reader reports no
                    # magnitude, so scan the fine ladder both directions.
                    steps = [s * mag for mag in (1, 2, 4, 8) for s in (1, -1)]
                plans.append({
                    "module": m,
                    "group": self._group_of(m),
                    "local": self._local_index(m),
                    "char_index": ci,
                    "steps": steps,
                    "absolute": True,
                    "cap": CHAR_OFFSET_LIMIT,
                    "targets": [ch],
                    "guards": guards,
                    "state": 0,
                    "best": 0,
                    "best_score": 0,
                    "label": f"g{self._group_of(m)} "
                             f"m{self._local_index(m)} c{ci} ({ch!r})",
                })
        if plans:
            self._cell_ladder(plans)

    def _p4_verify(self):
        """P4: repeatability plus folded border check (short forward hops)."""
        self.event("phase", "P4 repeatability + boundary check")
        for ch in (self.drum[0], self.drum[len(self.drum) // 2],
                   self.drum[-1]):
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
                if entry.char != ch or entry.condition == "double":
                    self._escalate(m, ch, f"boundary check: "
                                          f"{entry.char!r}/{entry.condition}")

    def _acceptance(self) -> tuple[bool, str]:
        if self.identity_persistent:
            mods = sorted({e["module"] for e in self.identity_persistent})
            return False, f"persistent identity/read problems on modules {mods}"
        self.event("phase", "acceptance: forward sweep over the whole drum")
        bad: dict[int, list] = {}
        for ch in self.drum:  # forward order: small consecutive hops
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
        """Per-phase seconds + budget deltas derived from phase marks."""
        out = []
        prev = {"elapsed": 0.0, "frames": 0, "vlmCalls": 0,
                "previews": 0, "persists": 0}
        for mark in self._phase_marks:
            out.append({
                "phase": mark["name"],
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
        self.group_widths = self._widths(status)
        snapshot = self.display.snapshot()
        settings = snapshot.get("settings", snapshot) if isinstance(snapshot, dict) else {}
        try:
            steps_per_rot = int(settings.get(
                "stepsPerRot", status.get("stepsPerRot", 2048)))
        except (TypeError, ValueError):
            steps_per_rot = 2048
        self.steps_per_char = max(
            1, round(steps_per_rot / max(1, len(self.drum)))) if self.drum else 1
        self._load_remote_offsets(settings)
        report: dict = {
            "contractVersion": SUPPORTED_CONTRACT,
            "exhaustive": self.exhaustive,
            "mode": self.mode,
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
                self._p1_coarse()
                self._p2_fine()
                self._p4_verify()
                ok, reason = self._acceptance()
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
