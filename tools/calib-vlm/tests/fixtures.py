"""Fakes for calib-vlm tests: no camera, display or network needed."""

from __future__ import annotations

import numpy as np

from calib.display import CalibError

from calib_vlm.calibrate import CHAR_OFFSET_LIMIT, PREVIEW_DELTA_MAX
from calib_vlm.reader import ModuleReading, Reading

CHARSET_37 = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHARSET_48 = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':?!.-/$@#%"


def matrix_to_csv(matrix) -> str:
    return ";".join(",".join(str(v) for v in row) for row in matrix)


class FakeDisplay:
    """Duck-typed Display with a simple physical model.

    A char offset o shifts the commanded character by round(o /
    steps_per_char) drum positions forward; a module offset re-anchors the
    magnet reference and so shifts the whole drum the OPPOSITE way (this
    mirrors the firmware: position = magnetPos + moduleOffset on magnet
    detection, then forward-only steps to charPosition). A non-multiple
    phase is a flap-seam condition. `remote_mod`/`remote_char` are the
    master-side offset tables exposed through /settings for remote groups.
    """

    def __init__(self, total: int = 4, groups: int = 1, charset: int = 37,
                 steps_per_rot: int = 2048, drum: str | None = None):
        self.total = total
        self.groups = max(1, groups)
        self.local = total if self.groups <= 1 else total // self.groups
        self.charset = charset
        self.drum = drum or (CHARSET_37 if charset == 37 else CHARSET_48)
        self.steps_per_rot = steps_per_rot
        self.spc = max(1, round(steps_per_rot / len(self.drum)))
        # Persisted (NVS) offsets.
        self.mod_off = [0] * self.local
        self.char_off: list[dict[int, int]] = [dict() for _ in range(self.local)]
        self.remote_mod = [[0] * 8 for _ in range(5)]
        self.remote_char = [[[0] * 48 for _ in range(8)] for _ in range(5)]
        # RAM-only preview residue layered on top of the persisted offsets.
        # `reload()` (firmware /api/calib/reload) drops it; `persist()` keeps
        # the value and drops the residue for that cell.
        self.res_mod = [0] * self.local
        self.res_char: list[dict[int, int]] = [dict() for _ in range(self.local)]
        # Remote volatile preview residue (fleet preview forwarded by the
        # master over ESP-NOW; RAM-only on the remote group).
        self.res_remote_mod = [[0] * 8 for _ in range(5)]
        self.res_remote_char = [[[0] * 48 for _ in range(8)]
                                for _ in range(5)]
        # Per-character mechanical landing error (steps): models a flap that
        # physically sits off its slot. A sub-pitch module trim can pull a
        # boundary flap back; `flap_window` is how far off-centre a landing
        # can be and still read clean (0 = only perfectly centred).
        self.mech_err: list[dict[int, int]] = [dict() for _ in range(self.local)]
        self.remote_mech = [[[0] * 48 for _ in range(8)] for _ in range(5)]
        self.flap_window = 0
        self.frame = " " * total
        self.fid = 0
        self.hold_active = False
        self.previews: list[tuple] = []
        self.persists: list[tuple] = []
        self.batches: list[list[dict]] = []
        self.restored: list = []
        self.reloads = 0

    # -- geometry -------------------------------------------------------------
    def widths(self) -> list[int]:
        if self.groups <= 1:
            return [self.total]
        widths = [self.local] * self.groups
        widths[-1] = self.total - self.local * (self.groups - 1)
        return widths

    def _group_local(self, i: int) -> tuple[int, int]:
        off = 0
        for g, width in enumerate(self.widths(), start=1):
            if i < off + width:
                return g, i - off
            off += width
        raise AssertionError(f"module {i} out of range")

    def offset_for(self, i: int, ci: int) -> int:
        """Net displayed-vs-commanded step offset (firmware sign).

        Uses the live (persisted + preview residue) values, like the
        firmware's `getLive*Offset()` accessors, plus the flap's mechanical
        landing error.
        """
        group, local = self._group_local(i)
        if group == 1:
            return ((self.char_off[local].get(ci, 0)
                     + self.res_char[local].get(ci, 0))
                    + self.mech_err[local].get(ci, 0)
                    - (self.mod_off[local] + self.res_mod[local]))
        return (self.remote_char[group - 2][local][ci]
                + self.remote_mech[group - 2][local][ci]
                + self.res_remote_char[group - 2][local][ci]
                - self.remote_mod[group - 2][local]
                - self.res_remote_mod[group - 2][local])

    # -- fault seeding --------------------------------------------------------
    def seed_module_error(self, i: int, steps: int):
        """Seed a whole-drum fault `steps` forward (+) or backward (-).

        Stored internal offsets are negated because a positive module
        offset shifts the drum backwards (see `offset_for`), so callers keep
        the intuitive "forward steps" meaning.
        """
        group, local = self._group_local(i)
        if group == 1:
            self.mod_off[local] -= steps
        else:
            self.remote_mod[group - 2][local] -= steps

    def seed_char_error(self, i: int, ci: int, steps: int):
        group, local = self._group_local(i)
        if group == 1:
            self.char_off[local][ci] = self.char_off[local].get(ci, 0) + steps
        else:
            self.remote_char[group - 2][local][ci] += steps

    def seed_flap_error(self, i: int, ci: int, steps: int):
        """Mechanical landing error for one flap (steps)."""
        group, local = self._group_local(i)
        if group == 1:
            self.mech_err[local][ci] = self.mech_err[local].get(ci, 0) + steps
        else:
            self.remote_mech[group - 2][local][ci] += steps

    # -- simulation -----------------------------------------------------------
    def displayed_char(self, i: int, cmd: str) -> str:
        if cmd not in self.drum:
            return cmd
        ci = self.drum.index(cmd)
        delta = round(self.offset_for(i, ci) / self.spc)
        return self.drum[(ci + delta) % len(self.drum)]

    def condition(self, i: int, cmd: str) -> str:
        if cmd not in self.drum:
            return "blank"
        ci = self.drum.index(cmd)
        total = self.offset_for(i, ci)
        delta = round(total / self.spc)
        if delta % len(self.drum) != 0:
            return "double"
        phase = total - delta * self.spc
        return "clean" if abs(phase) <= self.flap_window else "half"

    # -- Display interface ----------------------------------------------------
    def status(self) -> dict:
        rows = []
        for i in range(self.local):
            row = [0] * len(self.drum)
            for ci in range(len(row)):
                row[ci] = (self.char_off[i].get(ci, 0)
                           + self.res_char[i].get(ci, 0))
            rows.append(row)
        return {
            "contractVersion": 1, "schemaVersion": 1, "busy": False,
            "numModules": self.local, "totalModules": self.total,
            "groupCount": self.groups, "charset": self.charset,
            "drumOrder": self.drum, "displayOffset": 0,
            "moduleOffsets": [self.mod_off[i] + self.res_mod[i]
                              for i in range(self.local)],
            "charOffsets": rows, "holdActive": self.hold_active,
        }

    def contract(self) -> dict:
        return {"contractVersion": 1}

    def snapshot(self) -> dict:
        settings = {"stepsPerRot": self.steps_per_rot,
                    "rModOffs": matrix_to_csv(self.remote_mod)}
        for row in range(5):
            settings[f"rChrOff{row}"] = matrix_to_csv(self.remote_char[row])
        return {"settings": settings}

    def hold(self, active: bool) -> dict:
        self.hold_active = bool(active)
        return {"holdActive": self.hold_active}

    def show_and_settle(self, frame: str, dwell_ms: int = 800,
                        timeout_s: float | None = None, abort_flag=None,
                        pickup_grace_s: float = 3.0) -> dict:
        self.frame = frame
        self.fid += 1
        return {"frameId": self.fid, "fleetFrame": False}

    def wait_settled(self, timeout_s: float | None = None, poll_s: float = 0.5,
                     abort_flag=None) -> dict:
        return self.status()

    def preview(self, module: int, char_index: int, delta: int) -> dict:
        # Firmware contract (SplitFlapWebServer.cpp): a single preview is
        # non-zero; char cells are |delta| <= 32 and clamp the absolute
        # value to ±32, while module offsets accept up to a full revolution
        # and accumulate unconstrained.
        limit = PREVIEW_DELTA_MAX if char_index >= 0 else self.steps_per_rot
        if delta == 0 or abs(delta) > limit:
            raise CalibError("POST /api/calib/preview -> HTTP 400: "
                             f"Invalid delta (expected -{limit}..{limit}, "
                             "non-zero)")
        self.previews.append((module, char_index, delta))
        if char_index < 0:
            self.res_mod[module] += delta
        else:
            base = self.char_off[module].get(char_index, 0)
            live = base + self.res_char[module].get(char_index, 0) + delta
            live = max(-CHAR_OFFSET_LIMIT, min(CHAR_OFFSET_LIMIT, live))
            self.res_char[module][char_index] = live - base
        return {"type": "success"}

    def reload(self) -> dict:
        """Firmware /api/calib/reload: drop all RAM-only preview residue."""
        self.reloads += 1
        self.res_mod = [0] * self.local
        self.res_char = [dict() for _ in range(self.local)]
        self.res_remote_mod = [[0] * 8 for _ in range(5)]
        self.res_remote_char = [[[0] * 48 for _ in range(8)]
                                for _ in range(5)]
        return {"type": "success"}

    def preview_batch(self, nudges: list[dict]) -> dict:
        """Firmware /api/calib/preview-batch: several nudges, one pass.

        Scope 1 is the local controller; scope 2..6 is a remote group
        (forwarded by the master and applied RAM-only there).
        """
        self.batches.append(list(nudges))
        for nudge in nudges:
            scope = int(nudge.get("scope", 1))
            if scope == 1:
                self.preview(nudge["module"], nudge["charIndex"],
                             nudge["delta"])
            elif nudge["charIndex"] < 0:
                self.res_remote_mod[scope - 2][nudge["module"]] \
                    += nudge["delta"]
            else:
                row = scope - 2
                local = nudge["module"]
                ci = nudge["charIndex"]
                base = self.remote_char[row][local][ci]
                live = (base + self.res_remote_char[row][local][ci]
                        + nudge["delta"])
                live = max(-CHAR_OFFSET_LIMIT,
                           min(CHAR_OFFSET_LIMIT, live))
                self.res_remote_char[row][local][ci] = live - base
        return {"type": "success", "count": len(nudges)}

    def persist(self, scope, kind: str, value: int, module: int = 0,
                char_index: int = 0) -> dict:
        group = 1 if scope in ("local", 1, "1") else int(scope)
        if kind == "char":
            value = max(-CHAR_OFFSET_LIMIT,
                        min(CHAR_OFFSET_LIMIT, value))  # firmware clamps
        self.persists.append((group, kind, value, module, char_index))
        if group == 1:
            if kind == "module":
                self.mod_off[module] = value
                self.res_mod[module] = 0
            else:
                self.char_off[module][char_index] = value
                self.res_char[module].pop(char_index, None)
        else:
            row = group - 2
            if kind == "module":
                self.remote_mod[row][module] = value
                self.res_remote_mod[row][module] = 0
            elif kind == "char":
                self.remote_char[row][module][char_index] = value
                self.res_remote_char[row][module][char_index] = 0
        return {"type": "success"}

    def restore(self, snapshot: dict) -> dict:
        self.restored.append(snapshot)
        return {"type": "success"}


class FakeCamera:
    def __init__(self, *args, **kwargs):
        pass

    def open(self, quick: bool = False):
        pass

    def check_camera(self) -> dict:
        return {"ok": True}

    def capture(self):
        return np.zeros((60, 80, 3), dtype=np.uint8)

    def close(self):
        pass


class SimReader:
    """Perfect reader backed by a FakeDisplay; `frozen` forces a glyph."""

    def __init__(self, display: FakeDisplay, annotate: bool = False,
                 confidence: float = 0.99):
        self.display = display
        self.annotate = annotate
        self.confidence = confidence
        self.frozen: dict[int, str] = {}

    def read(self, jpeg: bytes, total: int, expected: str = "",
             charset: str = "", drum: str = "") -> Reading:
        frame = expected.ljust(total)[:total]
        modules = []
        for i in range(total):
            cmd = frame[i]
            if i in self.frozen:
                char, cond = self.frozen[i], "clean"
            else:
                char = self.display.displayed_char(i, cmd)
                cond = self.display.condition(i, cmd)
            modules.append(ModuleReading(i, char, cond, self.confidence,
                                         "vlm", cmd))
        return Reading(modules, raw_count=total)

    def error_reading(self, total: int, expected: str, problem: str) -> Reading:
        modules = [ModuleReading(i, "?", "unreadable", 0.0, "error",
                                 expected[i] if i < len(expected) else " ")
                   for i in range(total)]
        return Reading(modules, raw_count=0, warnings=[problem])
