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

    # -- reading one frame ----------------------------------------------------
    def _summary_line(self, reading: Reading) -> str:
        counts: dict[str, int] = {}
        for m in reading.modules:
            key = m.condition
            counts[key] = counts.get(key, 0) + 1
        return ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))

    def _show_read(self, frame: str, tag: str) -> tuple[dict, Reading]:
        self._guard_budgets(frame=True)
        info = self._show_and_settle(frame)
        if self.dwell_ms:
            time.sleep(self.dwell_ms / 1000.0)
        img = self.camera.capture()
        path = os.path.join(self.photo_dir, f"{tag}_f{info['frameId']}.png")
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
        rec = {"tag": tag, "frameId": info["frameId"], "frame": frame,
               "photo": os.path.basename(path)}
        rec.update(reading.as_dict())
        self.frames.append(rec)
        self.event("read", f"{tag}: {reading.text!r} "
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
        """Forward character distance from `seen` to `target` on the drum."""
        if not self.drum or seen not in self.drum or target not in self.drum:
            return None
        return (self.drum.index(target) - self.drum.index(seen)) % len(self.drum)

    def _identity_steps(self, seen: str, target: str) -> int | None:
        """Forward motor-step correction moving `seen` to `target`."""
        chars = self._drum_delta(seen, target)
        if chars is None:
            return None
        return chars * self.steps_per_char

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
        self.event("tune", f"{state} g{group} m{local} c{char_index} "
                           f"target {target!r}: "
                           f"{before.char!r}/{before.condition} -> "
                           f"{after.char!r}/{after.condition}",
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
        Module cells are unconstrained; local previews are chunked because
        the firmware rejects a single |delta| > 32.
        """
        target_value = self.live(key) + delta
        if char_index >= 0 and abs(target_value) > CHAR_OFFSET_LIMIT:
            return False
        if group == 1:
            for chunk in _preview_chunks(delta):
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

        Uses forward drum-order deltas; local modules via volatile preview
        then absolute persist, remote modules via persist-verify."""
        group = self._group_of(module)
        local = self._local_index(module)
        key = self._ensure_cell(group, local, char_index)
        kind = self._kind(char_index)
        tag = f"tune_g{group}m{local}c{char_index}"
        before: ModuleReading | None = None
        for _ in range(MAX_TUNE_ITER):
            _, reading = self._show_read(show_frame, tag)
            entry = reading.modules[module]
            if before is None:
                before = entry
            if entry.char == target and entry.condition == "clean":
                self._commit_local(group, key, kind, char_index)
                return self._record_delta(module, char_index, target, before,
                                          entry, fixed=True, delta=0)
            if self.mode == "dry-run" or (self.mode == "preview" and group != 1):
                delta = self._identity_steps(entry.char, target)
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
            delta = self._identity_steps(entry.char, target)
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

    def _p1_coarse(self):
        glyphs = self._coarse_glyphs()
        coverage = len(glyphs) / max(1, len(self.drum))
        self.event("phase", f"P1 coarse identity / whole-drum offsets "
                            f"({len(glyphs)} uniform glyphs, "
                            f"{coverage:.0%} of the drum)")
        votes: dict[int, list[str]] = {}
        for glyph in glyphs:
            _, reading = self._show_read(glyph * self.total, f"p1_{ord(glyph)}")
            for i, entry in enumerate(reading.modules):
                if self._trusted(entry) and entry.char != glyph:
                    votes.setdefault(i, []).append(glyph)
        for module, glyph_list in sorted(votes.items()):
            if len(glyph_list) >= COARSE_VOTE_MIN:
                self._tune_identity(module, -1, glyph_list[0],
                                    glyph_list[0] * self.total)
            else:
                self._tune_identity(module, self.drum.index(glyph_list[0]),
                                    glyph_list[0], glyph_list[0] * self.total)
        # Verify pass: same coarse-vs-char vote split as the first pass, so
        # a surviving single-glyph fault is never "fixed" by shifting the
        # whole drum (which would break the other characters on it).
        remaining: dict[int, list[str]] = {}
        for glyph in glyphs:
            _, reading = self._show_read(glyph * self.total,
                                         f"p1v_{ord(glyph)}")
            for i, entry in enumerate(reading.modules):
                if self._trusted(entry) and entry.char != glyph:
                    remaining.setdefault(i, []).append(glyph)
        for module, glyph_list in sorted(remaining.items()):
            if len(glyph_list) >= COARSE_VOTE_MIN:
                self._tune_identity(module, -1, glyph_list[0],
                                    glyph_list[0] * self.total)
            else:
                self._tune_identity(module, self.drum.index(glyph_list[0]),
                                    glyph_list[0], glyph_list[0] * self.total)
        _, final = self._show_read("H" * self.total, "p1_check")
        for i, entry in enumerate(final.modules):
            if not self._trusted(entry):
                if entry.condition == "unreadable":
                    self._escalate(i, "H", "cannot read module after P1")
                continue
            if entry.char != "H":
                self._escalate(i, "H", f"reads {entry.char!r} after P1")

    def _p2_fine(self):
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
        for k in range(0, n - 1, 7):
            frame = "".join(self.drum[k + (i % 2)] for i in range(self.total))
            _, reading = self._show_read(frame, f"p3_{k}")
            for i, entry in enumerate(reading.modules):
                expected = frame[i]
                if expected == " " or not self._trusted(entry):
                    continue
                ci = self.drum.index(expected)
                if entry.char != expected:
                    self._tune_identity(i, ci, expected,
                                        expected * self.total)
                elif entry.condition == "double":
                    self._tune_alignment(i, ci, expected,
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
