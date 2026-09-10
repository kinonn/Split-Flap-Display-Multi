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

    The step offset o on a module shifts the displayed character by
    round(o / steps_per_char) drum positions; a non-multiple phase is a
    flap-seam condition. `remote_mod`/`remote_char` are the master-side
    offset tables exposed through /settings for remote groups.
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
        # Firmware-visible live state: local offsets + remote tables.
        self.mod_off = [0] * self.local
        self.char_off: list[dict[int, int]] = [dict() for _ in range(self.local)]
        self.remote_mod = [[0] * 8 for _ in range(5)]
        self.remote_char = [[[0] * 48 for _ in range(8)] for _ in range(5)]
        self.frame = " " * total
        self.fid = 0
        self.hold_active = False
        self.previews: list[tuple] = []
        self.persists: list[tuple] = []
        self.restored: list = []

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
        group, local = self._group_local(i)
        if group == 1:
            return self.mod_off[local] + self.char_off[local].get(ci, 0)
        return (self.remote_mod[group - 2][local]
                + self.remote_char[group - 2][local][ci])

    # -- fault seeding --------------------------------------------------------
    def seed_module_error(self, i: int, steps: int):
        group, local = self._group_local(i)
        if group == 1:
            self.mod_off[local] += steps
        else:
            self.remote_mod[group - 2][local] += steps

    def seed_char_error(self, i: int, ci: int, steps: int):
        group, local = self._group_local(i)
        if group == 1:
            self.char_off[local][ci] = self.char_off[local].get(ci, 0) + steps
        else:
            self.remote_char[group - 2][local][ci] += steps

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
        return "clean" if phase == 0 else "half"

    # -- Display interface ----------------------------------------------------
    def status(self) -> dict:
        rows = []
        for i in range(self.local):
            row = [0] * len(self.drum)
            for ci, value in self.char_off[i].items():
                if 0 <= ci < len(row):
                    row[ci] = value
            rows.append(row)
        return {
            "contractVersion": 1, "schemaVersion": 1, "busy": False,
            "numModules": self.local, "totalModules": self.total,
            "groupCount": self.groups, "charset": self.charset,
            "drumOrder": self.drum, "displayOffset": 0,
            "moduleOffsets": list(self.mod_off), "charOffsets": rows,
            "holdActive": self.hold_active,
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
        # non-zero, |delta| <= 32, and clamps the char cell's absolute
        # value to ±32 (module offsets accumulate unconstrained).
        if delta == 0 or abs(delta) > PREVIEW_DELTA_MAX:
            raise CalibError("POST /api/calib/preview -> HTTP 400: "
                             "Invalid delta (expected -32..32, non-zero)")
        self.previews.append((module, char_index, delta))
        if char_index < 0:
            self.mod_off[module] += delta
        else:
            self.char_off[module][char_index] = max(
                -CHAR_OFFSET_LIMIT,
                min(CHAR_OFFSET_LIMIT,
                    self.char_off[module].get(char_index, 0) + delta))
        return {"type": "success"}

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
            else:
                self.char_off[module][char_index] = value
        else:
            row = group - 2
            if kind == "module":
                self.remote_mod[row][module] = value
            elif kind == "char":
                self.remote_char[row][module][char_index] = value
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
