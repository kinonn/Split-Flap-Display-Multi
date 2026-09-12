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
import math
import os
import time

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
# Deltas tried per suspect cell (motor steps), coarse first.
TRY_DELTAS = (4, -4, 2, -2, 8, -8, 1, -1)
CONDITION_COST = {"clean": 0, "blank": 0, "half": 1, "double": 2,
                  "stuck": 3, "unreadable": 4}
# Uniform glyphs used to split coarse (whole-drum) from per-char faults.
# P1 always reads at least COARSE_COVERAGE of the drum, starting from
# these easy-to-read glyphs and spreading across the rest.
COARSE_GLYPHS = ("E", "H", "O", "0", "-")
COARSE_COVERAGE = 0.25
ACCEPTANCE_GLYPHS = ("E", "H")
# A module wrong on this many uniform frames is treated as coarse
# (whole-drum homing); a single-glyph fault is a per-char cell.
COARSE_VOTE_MIN = 2
# A module misaligned (half/double, identity still right) on this many
# uniform frames is a whole-drum phase fault: tune the module cell once
# instead of scattering per-char alignment searches across the drum.
ALIGN_VOTE_MIN = 2
# Sub-pitch module-offset trims tried BEFORE per-character cells: a module
# offset slides every character's landing by the same steps, so a fraction
# of one flap pulls characters sitting just past their flap boundary back
# onto their own flap without moving well-centred characters. Multiples of
# a full character stay on the coarse `_identity_steps` path.
MODULE_TRIM_FRACTIONS = (0.75, 0.5, 0.25, 0.125)
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


class VlmCalibrator:
    def __init__(self, display, camera, reader, photo_dir: str,
                 dwell_ms: int = 800, timeout_s: float = 60.0,
                 min_confidence: float = 0.6, exhaustive: bool = False,
                 mode: str = "full", on_event=None, max_seconds: float = 3600.0):
        if mode not in ("dry-run", "preview", "full"):
            raise ValueError("mode must be dry-run, preview or full")
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
        self._t0 = time.monotonic()
        os.makedirs(photo_dir, exist_ok=True)

    # -- events / abort -------------------------------------------------------
    def event(self, kind: str, text: str, photo: str | None = None,
              detail: dict | None = None):
        evt: dict = {"t": time.strftime("%H:%M:%S"), "kind": kind,
                     "text": text, "photo": photo}
        if detail is not None:
            evt["detail"] = detail
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
        # Account the real VLM round trips: a read may re-ask once, and
        # failed parses still consumed provider calls.
        self._charge_vlm(max(1, getattr(self.reader, "last_calls", 1)))
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
                and entry.char not in ("", "?", None)
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

    def _coarse_glyphs(self) -> list[str]:
        """Uniform P1 test glyphs: at least COARSE_COVERAGE of the drum.

        Coverage matters: a whole-drum offset is visible on any glyph, but
        a per-character fault only shows on the character it belongs to,
        so P1 samples the easy glyphs plus an even spread across the rest.
        Space is skipped (blank flaps carry no identity information).
        """
        non_space = [c for c in self.drum if c != " "]
        if not non_space:
            return []
        target = max(len(COARSE_GLYPHS),
                     math.ceil(len(self.drum) * COARSE_COVERAGE))
        target = min(target, len(non_space))
        glyphs: list[str] = []
        for glyph in COARSE_GLYPHS:
            if glyph in non_space and glyph not in glyphs:
                glyphs.append(glyph)
        for i in range(target):
            if len(glyphs) >= target:
                break
            glyph = non_space[(i * len(non_space)) // target]
            if glyph not in glyphs:
                glyphs.append(glyph)
        return glyphs

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

    def _batch_nudge(self, nudges: list[tuple[int, int, int]]):
        """Apply module-cell nudges [(group, local, delta), ...] in one pass.

        Group 1 is applied locally; groups 2..6 are forwarded by the master
        over ESP-NOW and applied RAM-only on each remote (the master's busy
        fence covers their homing via preview acks). N modules across the
        fleet cost one homing pass, not N. Falls back to serial single
        previews on older firmware (local group only).
        """
        if not nudges:
            return
        self._guard_budgets(preview=True)
        batch = getattr(self.display, "preview_batch", None)
        by_scope: dict[int, list[tuple[int, int]]] = {}
        for group, local, delta in nudges:
            by_scope.setdefault(group, []).append((local, delta))
        for group, items in sorted(by_scope.items()):
            if batch is not None:
                batch([{"scope": group, "module": local, "charIndex": -1,
                        "delta": delta} for local, delta in items])
            elif group == 1:
                for local, delta in items:
                    self.display.preview(local, -1, delta)
            else:
                continue  # remote preview needs the fleet endpoint
            if group == 1:
                for local, delta in items:
                    key = (1, local, -1)
                    self.residue[key] = self.residue.get(key, 0) + delta
        self.previews += len(nudges)
        self._wait_settled(self.timeout_s)

    def _module_trim(self, votes: dict[int, list[str]],
                     seen: dict[int, dict[str, str]],
                     glyphs: list[str]) -> set[int]:
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
            plans.append({
                "module": module,
                "group": self._group_of(module),
                "local": self._local_index(module),
                "direction": -1 if errors[0] > 0 else 1,
                "targets": wrong[:MODULE_TRIM_TARGETS],
                "guards": guards[:MODULE_TRIM_GUARDS],
                "state": 0,
                "best": 0,
                "best_score": 0,
            })
        if not plans:
            return set()
        # Seed every cell's base BEFORE nudging: the applied candidates must
        # land in the residue so a commit persists overlay + residue.
        for p in plans:
            self._ensure_cell(p["group"], p["local"], -1)
        by_module = {p["module"]: p for p in plans}
        rounds = [max(1, int(round(self.steps_per_char * f)))
                  for f in MODULE_TRIM_FRACTIONS]
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
                _, reading = self._show_read(frame_for(index), f"trim_t{index}")
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
                                             f"trim_g{index}")
                for p in plans:
                    if index >= len(p["guards"]):
                        continue
                    entry = reading.modules[p["module"]]
                    if (not self._trusted(entry)
                            or entry.char != p["guards"][index]
                            or entry.condition not in ("clean", "blank")):
                        scores[p["module"]] -= 6
            return scores

        self.event("phase", f"P1 module trim ({len(plans)} modules, "
                            f"sub-{self.steps_per_char} steps, parallel)")
        for magnitude in rounds:
            apply = []
            for p in plans:
                candidate = p["best"] + p["direction"] * magnitude
                if abs(candidate) >= self.steps_per_char:
                    candidate = p["best"]  # stay strictly sub-pitch
                p["candidate"] = candidate
                diff = candidate - p["state"]
                p["state"] = candidate
                if diff:
                    apply.append((p["group"], p["local"], diff))
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
                        revert.append((p["group"], p["local"], diff))
            self._batch_nudge(revert)

        improved = set()
        for p in plans:
            diff = p["best"] - p["state"]
            if diff:
                self._batch_nudge([(p["group"], p["local"], diff)])
                p["state"] = p["best"]
            if p["best"] != 0 and p["best_score"] > 0:
                key = self._ensure_cell(p["group"], p["local"], -1)
                if p["group"] == 1:
                    self._commit_local(1, key, "module", -1)
                elif self.mode == "full":
                    # Remote previews are RAM-only: persist the absolute
                    # winner (base + trim) so the fix survives the revert.
                    # Unreachable in preview (remote modules are filtered
                    # out of plans above); guarded anyway.
                    base = self._read_cell(p["group"], p["local"], -1)
                    self._guard_budgets()
                    self.display.persist(p["group"], "module",
                                         base + p["best"], p["local"], 0)
                    self.persists += 1
                    self._wait_settled(self.timeout_s)
                    self.overlay[key] = base + p["best"]
                    self.residue[key] = 0
                self.module_fixed.add(p["module"])
                self._record_delta(p["module"], -1, p["targets"][0], None,
                                   None, fixed=True, delta=p["best"])
                improved.add(p["module"])
                self.event("tune", f"module trim g{p['group']} m{p['local']}: "
                                   f"{p['best']:+d} steps "
                                   f"(score {p['best_score']})")
        return improved

    def _p1_coarse(self):
        glyphs = self._coarse_glyphs()
        coverage = len(glyphs) / max(1, len(self.drum))
        self.event("phase", f"P1 coarse identity / whole-drum offsets "
                            f"({len(glyphs)} uniform glyphs, "
                            f"{coverage:.0%} of the drum)")
        votes: dict[int, list[str]] = {}
        seen: dict[int, dict[str, str]] = {}
        align_votes: dict[int, list[str]] = {}
        for glyph in glyphs:
            _, reading = self._show_read(glyph * self.total, f"p1_{ord(glyph)}")
            for i, entry in enumerate(reading.modules):
                if not self._trusted(entry):
                    continue
                if entry.char != glyph:
                    votes.setdefault(i, []).append(glyph)
                    seen.setdefault(i, {})[glyph] = entry.char
                elif entry.condition in ("half", "double"):
                    # Identity right but flap off-phase on this glyph: a
                    # candidate whole-drum alignment fault (module cell),
                    # confirmed below by the vote count.
                    align_votes.setdefault(i, []).append(glyph)
        # Sub-pitch module trim before any per-char cell: a module wrong on
        # only a few glyphs is usually a boundary/phase problem, not several
        # independent broken characters. Runs in parallel across modules.
        trimmed = self._module_trim(votes, seen, glyphs)
        for module, glyph_list in sorted(votes.items()):
            if module in trimmed:
                continue
            if len(glyph_list) >= COARSE_VOTE_MIN:
                self._tune_identity(module, -1, glyph_list[0],
                                    glyph_list[0] * self.total)
                self.module_fixed.add(module)
            else:
                self._tune_identity(module, self.drum.index(glyph_list[0]),
                                    glyph_list[0], glyph_list[0] * self.total)
        # Module-alignment pass (manual process, layer 1): a module whose
        # flaps sit off-phase on several glyphs gets ONE module-cell
        # alignment search, not a per-char search per glyph. Modules
        # already identity-tuned above are skipped (their phase moved).
        identity_tuned = set(votes)
        for module, glyph_list in sorted(align_votes.items()):
            if module in identity_tuned or len(glyph_list) < ALIGN_VOTE_MIN:
                continue
            self._tune_alignment(module, -1, glyph_list[0],
                                 glyph_list[0] * self.total)
            self.module_fixed.add(module)
        # Verify pass: same coarse-vs-char vote split as the first pass, so
        # a surviving single-glyph fault is never "fixed" by shifting the
        # whole drum (which would break the other characters on it).
        # Modules fixed at module level above are re-checked, not re-tuned
        # here: P2/P3 only see them again if the verify still shows them.
        remaining: dict[int, list[str]] = {}
        remaining_align: dict[int, list[str]] = {}
        for glyph in glyphs:
            _, reading = self._show_read(glyph * self.total,
                                         f"p1v_{ord(glyph)}")
            for i, entry in enumerate(reading.modules):
                if not self._trusted(entry):
                    continue
                if entry.char != glyph:
                    remaining.setdefault(i, []).append(glyph)
                elif entry.condition in ("half", "double"):
                    remaining_align.setdefault(i, []).append(glyph)
        for module, glyph_list in sorted(remaining.items()):
            if len(glyph_list) >= COARSE_VOTE_MIN:
                self._tune_identity(module, -1, glyph_list[0],
                                    glyph_list[0] * self.total)
                self.module_fixed.add(module)
            else:
                self._tune_identity(module, self.drum.index(glyph_list[0]),
                                    glyph_list[0], glyph_list[0] * self.total)
        for module, glyph_list in sorted(remaining_align.items()):
            if (module in remaining or module in identity_tuned
                    or len(glyph_list) < ALIGN_VOTE_MIN):
                continue
            self._tune_alignment(module, -1, glyph_list[0],
                                 glyph_list[0] * self.total)
            self.module_fixed.add(module)
        _, final = self._show_read("H" * self.total, "p1_check")
        for i, entry in enumerate(final.modules):
            if not self._trusted(entry):
                if entry.condition == "unreadable":
                    self._escalate(i, "H", "cannot read module after P1")
                continue
            if entry.char != "H":
                self._escalate(i, "H", f"reads {entry.char!r} after P1")

    def _recheck_module_fixed(self, suspects: dict[tuple[int, int], str],
                              align: dict[tuple[int, int], str]) -> None:
        """Re-verify per-char suspects on module-fixed modules in place.

        A module-cell fix moves every character on that drum, so a suspect
        collected before (or during) the fix may already be gone. For each
        suspect frame on a fixed module, re-show the uniform suspect frame
        and drop suspects the fresh reading no longer confirms. Mutates
        both dicts; modules with no surviving suspects leave module_fixed.
        """
        by_module: dict[int, set[str]] = {}
        for (module, _ci), expected in list(suspects.items()):
            if module in self.module_fixed:
                by_module.setdefault(module, set()).add(expected)
        for (module, _ci), expected in list(align.items()):
            if module in self.module_fixed:
                by_module.setdefault(module, set()).add(expected)
        # Fixed modules with no suspects at all need no re-check: the
        # sweep already verified them clean.
        for module in list(self.module_fixed):
            if module not in by_module:
                self.module_fixed.discard(module)
        for module in sorted(by_module):
            for glyph in sorted(by_module[module]):
                _, reading = self._show_read(glyph * self.total,
                                             f"recheck_m{module}_{ord(glyph)}")
                entry = reading.modules[module]
                if not self._trusted(entry):
                    continue  # keep suspects; an unreadable re-check proves nothing
                ci = self.drum.index(glyph) if glyph in self.drum else -1
                if entry.char == glyph and entry.condition in ("clean", "blank"):
                    suspects.pop((module, ci), None)
                    align.pop((module, ci), None)
                elif entry.char != glyph:
                    # Still the wrong glyph: keep the identity suspect, drop
                    # any alignment suspect for the same cell (identity owns it).
                    align.pop((module, ci), None)
                # else: right glyph, still half/double -> keep align suspect.
            if not any(m == module for (m, _c) in suspects) \
                    and not any(m == module for (m, _c) in align):
                self.module_fixed.discard(module)

    def _p2_fine(self):
        # Manual process, layer 2 (outliers only): modules already fixed
        # at module level in P1 are re-checked before any char cell is
        # touched — a stale module offset moves the whole drum.
        label = "exhaustive per-character" if self.exhaustive else "sampled"
        self.event("phase", f"P2 fine sweeps ({label})")
        n = len(self.drum)
        stride = 6
        offsets = range(stride) if self.exhaustive else (0, stride // 2)
        prev: Reading | None = None
        stuck_votes: dict[int, int] = {}
        comparisons = 0
        id_suspects: dict[tuple[int, int], str] = {}
        align_suspects: dict[tuple[int, int], str] = {}
        for offset in offsets:
            for k in range(offset, n, stride):
                self._guard_budgets()
                frame = "".join(self.drum[(k + i) % n]
                                for i in range(self.total))
                _, reading = self._show_read(frame, f"p2_{k}")
                if prev is not None:
                    comparisons += 1
                    for i, (a, b) in enumerate(zip(prev.modules,
                                                   reading.modules)):
                        if (self._trusted(a) and self._trusted(b)
                                and a.char == b.char and a.char != " "):
                            stuck_votes[i] = stuck_votes.get(i, 0) + 1
                prev = reading
                for i, entry in enumerate(reading.modules):
                    expected = frame[i]
                    if expected == " " or not self._trusted(entry):
                        continue
                    ci = self.drum.index(expected)
                    if entry.char != expected:
                        id_suspects[(i, ci)] = expected
                    elif entry.condition in ("half", "double"):
                        align_suspects[(i, ci)] = expected
            self.sweeps += 1
        # Outliers only: re-verify suspects on module-fixed modules
        # before touching any char cell.
        self._recheck_module_fixed(id_suspects, align_suspects)
        for (module, ci), expected in sorted(id_suspects.items()):
            self._tune_identity(module, ci, expected, expected * self.total)
        for (module, ci), expected in sorted(align_suspects.items()):
            self._tune_alignment(module, ci, expected, expected * self.total)
        for module, count in sorted(stuck_votes.items()):
            if comparisons and count * 2 >= comparisons:
                self._escalate(module, "?",
                               "reads the same glyph across different "
                               "commanded frames (stuck?)")

    def _p3_boundaries(self):
        self.event("phase", "P3 drum-neighbour boundaries")
        n = len(self.drum)
        id_suspects: dict[tuple[int, int], str] = {}
        align_suspects: dict[tuple[int, int], str] = {}
        for k in range(0, n - 1, 7):
            frame = "".join(self.drum[k + (i % 2)] for i in range(self.total))
            _, reading = self._show_read(frame, f"p3_{k}")
            for i, entry in enumerate(reading.modules):
                expected = frame[i]
                if expected == " " or not self._trusted(entry):
                    continue
                ci = self.drum.index(expected)
                if entry.char != expected:
                    id_suspects[(i, ci)] = expected
                elif entry.condition == "double":
                    align_suspects[(i, ci)] = expected
        # Same outlier rule as P2: module-fixed modules re-verify first.
        self._recheck_module_fixed(id_suspects, align_suspects)
        for (module, ci), expected in sorted(id_suspects.items()):
            self._tune_identity(module, ci, expected,
                                expected * self.total)
        for (module, ci), expected in sorted(align_suspects.items()):
            self._tune_alignment(module, ci, expected,
                                 expected * self.total)

    def _p4_repeatability(self):
        self.event("phase", "P4 repeatability (3x same frame)")
        readings = []
        for r in range(3):
            _, reading = self._show_read("H" * self.total, f"p4_{r}")
            readings.append(reading)
        for r in range(1, 3):
            for i, (a, b) in enumerate(zip(readings[0].modules,
                                           readings[r].modules)):
                if a.char != b.char or a.condition != b.condition:
                    self._escalate(
                        i, "H", f"unstable reads across repeats "
                                f"({a.char!r}/{a.condition} vs "
                                f"{b.char!r}/{b.condition})")

    def _acceptance(self) -> tuple[bool, str]:
        if self.identity_persistent:
            mods = sorted({e["module"] for e in self.identity_persistent})
            return False, f"persistent identity/read problems on modules {mods}"
        for glyph in ACCEPTANCE_GLYPHS:
            _, reading = self._show_read(glyph * self.total,
                                         f"accept_{glyph}")
            bad = []
            for i, entry in enumerate(reading.modules):
                if not self._trusted(entry):
                    bad.append((i, "unreadable"))
                elif entry.char != glyph:
                    bad.append((i, f"shows {entry.char!r}"))
                elif entry.condition != "clean":
                    bad.append((i, entry.condition))
            if bad:
                return False, f"acceptance failed on modules {bad}"
        return True, "acceptance frames read clean and correct on all modules"

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

    # -- main -----------------------------------------------------------------
    def run(self) -> dict:
        self._t0 = time.monotonic()
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
            self._p1_coarse()
            self._p2_fine()
            self._p3_boundaries()
            self._p4_repeatability()
            ok, reason = self._acceptance()
            if ok and self.mode != "full":
                ok = False
                reason = (f"{self.mode} mode: acceptance passed but no "
                          "offsets were persisted")
            report["result"] = "converged" if ok else "needs-human"
            report["reason"] = reason
        except CalibError as exc:
            report["reason"] = str(exc)
            if self.aborted:
                report["reason"] = "aborted by user"
            try:
                self.display.restore(snapshot)
            except CalibError as exc2:
                report["reason"] += f" | rollback failed: {exc2}"
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
            with open(os.path.join(self.photo_dir, "report.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
            self.report = report
            self.event("done", f"{report['result']}: {report['reason']}")
        return report
