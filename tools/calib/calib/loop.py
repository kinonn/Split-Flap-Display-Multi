"""P0->P4 calibration state machine (see tools/calib-agent/PRODUCTION.md).

Runs against duck-typed `display` (calib/display.py Display or a test
fake) and `camera` (calib/camera.py Camera or a test fake). Group 1 uses
preview-then-persist; remote groups (master-only access, no remote
preview in firmware) use persist-verify-revert per cell.
"""

from __future__ import annotations

import json
import os
import time

import cv2

from . import vision
from .display import CalibError

SUPPORTED_CONTRACT = 1
# Wear/time budgets.
MAX_FULL_SWEEPS = 3
MAX_PREVIEWS = 200
MAX_PERSISTS = 400
# Deltas tried per suspect cell (motor steps), coarse first.
TRY_DELTAS = (4, -4, 2, -2, 8, -8, 1, -1)
# Consensus identity settings.
IDENTITY_MIN_MODULES = 3  # abstain below this (cannot isolate blame)
IDENTITY_P1_VOTES = 2  # uniform frames flagging a module before re-verify
IDENTITY_MAX_VERIFY_GLYPHS = 8  # cap on extra uniform re-verify shows
VERDICT_COST = {"ok": 0, "half": 1, "double": 2}


def _cost(score: dict) -> tuple:
    return (VERDICT_COST[score["verdict"]], score["seam_strength"])


class Calibrator:
    def __init__(self, display, camera, photo_dir: str, dwell_ms: int = 800,
                 timeout_s: float = 60.0, max_phase: int = 4,
                 identity_thresh: float = vision.IDENTITY_THRESH,
                 relearn_templates: bool = False):
        self.display = display
        self.camera = camera
        self.photo_dir = photo_dir
        self.dwell_ms = dwell_ms
        self.timeout_s = timeout_s
        self.max_phase = max_phase
        self.identity_thresh = identity_thresh
        self.relearn_templates = relearn_templates
        self.frames: list[dict] = []
        self.deltas: list[dict] = []
        self.identity: list[dict] = []  # every identity check
        self.identity_persistent: list[dict] = []  # re-verified wrong glyphs
        self.templates: dict[str, object] = {}  # glyph -> template image
        self.template_source: str | None = None
        self.bank_samples: dict[str, list] = {}  # bootstrap candidates
        self.overlay: dict[tuple[int, int, int], int] = {}
        self.residue: dict[tuple[int, int, int], int] = {}
        self.previews = 0
        self.persists = 0
        self.sweeps = 0
        os.makedirs(photo_dir, exist_ok=True)

    # -- primitives ---------------------------------------------------------
    def shoot(self, frame: str, tag: str) -> dict:
        """Show a frame, settle, photograph, split into crops. Returns record."""
        info = self.display.show_and_settle(frame, self.dwell_ms, self.timeout_s)
        time.sleep(self.dwell_ms / 1000.0)
        img = self.camera.capture()
        path = os.path.join(self.photo_dir, f"{tag}_f{info['frameId']}.png")
        cv2.imwrite(path, img)
        gray = vision.to_gray(img)
        rec = {"tag": tag, "frameId": info["frameId"], "frame": frame,
               "photo": path, "fleetFrame": info["fleetFrame"]}
        self.frames.append(rec)
        rec["crops"] = vision.split_crops(gray, self.total)
        return rec

    def scores(self, rec: dict) -> list[dict]:
        return [vision.score_crop(c) for c in rec["crops"]]

    def consensus(self, rec: dict, glyph: str) -> list[int]:
        """Identity check on a uniform frame: crops scoring ok should all
        look alike. Misaligned crops are excluded (seam scoring owns them);
        a clean crop disagreeing with clean peers is the wrong glyph."""
        ok_idx = [i for i, c in enumerate(rec["crops"])
                  if vision.score_crop(c)["verdict"] == "ok"]
        if len(ok_idx) < IDENTITY_MIN_MODULES:
            self.identity.append({"tag": rec["tag"], "frameId": rec["frameId"],
                                  "glyph": glyph, "outliers": [],
                                  "note": "abstained (<3 clean crops)"})
            return []
        sub = [rec["crops"][i] for i in ok_idx]
        out = [ok_idx[k] for k in vision.consensus_outliers(sub, self.identity_thresh)]
        self.identity.append({"tag": rec["tag"], "frameId": rec["frameId"],
                              "glyph": glyph, "method": "consensus", "outliers": out})
        return out

    def absolute(self, rec: dict, glyph: str) -> list[int]:
        """Absolute identity: each clean crop's top template hit must be the
        commanded glyph. Catches what consensus cannot: all modules showing
        the same WRONG glyph, and fleets too small for consensus."""
        if glyph not in self.templates:
            self.identity.append({"tag": rec["tag"], "frameId": rec["frameId"],
                                  "glyph": glyph, "method": "template",
                                  "outliers": [], "note": "no template for glyph"})
            return []
        out = []
        for i, crop in enumerate(rec["crops"]):
            if vision.score_crop(crop)["verdict"] != "ok":
                continue
            if float(vision.to_gray(crop).std()) < vision.IDENTITY_MIN_STD:
                continue  # blank/flat crop carries no identity information
            ranked = vision.identify(crop, self.templates)
            if not ranked or ranked[0][0] != glyph or ranked[0][1] < vision.IDENTITY_ABSOLUTE_MIN:
                out.append(i)
        self.identity.append({"tag": rec["tag"], "frameId": rec["frameId"],
                              "glyph": glyph, "method": "template", "outliers": out})
        return out

    def _verify_uniform(self, glyph: str, suspects: set[int], stage: str):
        """Re-show one uniform frame; suspects still failing EITHER method
        (consensus or absolute template) become persistent escalations."""
        rec = self.shoot(glyph * self.total, f"id_verify_{ord(glyph)}")
        bad = set(self.consensus(rec, glyph)) | set(self.absolute(rec, glyph))
        for module in sorted(suspects & bad):
            self.identity_persistent.append(
                {"module": module, "glyph": glyph, "stage": stage,
                 "note": "clean crop disagrees with peers/template across frames"})

    def _templates_dir(self) -> str:
        return os.path.join(self.photo_dir, "templates")

    def _load_bank(self):
        import json

        manifest_path = os.path.join(self._templates_dir(), "manifest.json")
        if self.relearn_templates:
            return
        try:
            bank, manifest = vision.load_template_bank(self._templates_dir())
        except (OSError, ValueError, KeyError):
            return
        if bank:
            self.templates = bank
            self.template_source = manifest.get("source", "stored")

    def _save_bank(self, source: str):
        import time as _time

        vision.save_template_bank(
            self.templates,
            {"source": source, "created": _time.strftime("%Y-%m-%dT%H:%M:%S")},
            self._templates_dir(),
        )
        self.template_source = source

    def _bootstrap_missing(self) -> bool:
        """Build templates for glyphs seen with >=2 trusted samples.

        Trusted = clean crop on a module no method flagged. A stored
        (golden) bank is never auto-extended: a systematically shifted run
        would poison it with wrong-labeled templates. Golden banks only
        change via --relearn-templates after verified calibration.
        Returns True when the bank changed."""
        if self.template_source is not None:
            return False
        added = False
        for glyph, crops in self.bank_samples.items():
            if glyph in self.templates or len(crops) < 2:
                continue
            built = vision.build_template_bank({glyph: crops})
            if built:
                self.templates.update(built)
                added = True
        return added

    # -- tuning helpers ------------------------------------------------------
    def _group_of(self, module: int) -> int:
        """1-based group number for a fleet-wide module index."""
        off = 0
        for g, width in enumerate(self.group_widths, start=1):
            if module < off + width:
                return g
            off += width
        return 1

    def _local_index(self, module: int) -> int:
        off = sum(self.group_widths[: self._group_of(module) - 1])
        return module - off

    def _tune_cell(self, module: int, char_index: int, show_frame: str) -> dict:
        """Tune one cell; returns {kept, after}.

        Tracks assumed live values locally: `overlay` holds persisted
        values (module/display offsets readable from status, char cells
        start at 0), `residue` sums volatile preview deltas on top.
        Persists write absolute values, which also discards residues.
        """
        group = self._group_of(module)
        local = self._local_index(module)
        key = (group, local, char_index)
        if key not in self.overlay:
            self.overlay[key] = self._read_cell(group, local, char_index)
            self.residue[key] = 0
        kind = "char" if char_index >= 0 else "module"

        def live() -> int:
            return self.overlay[key] + self.residue[key]

        before = self.scores(self.shoot(show_frame, f"tune_g{group}m{local}c{char_index}"))[module]
        cost_before = _cost(before)
        old = self.overlay[key]
        if self.max_phase < 2:
            # Phase 1 is read-only: measure and propose, touch nothing.
            self.deltas.append({"scope": group, "module": local, "charIndex": char_index,
                                "old": old, "new": old, "proposal": True,
                                "cost_before": list(cost_before),
                                "cost_after": list(cost_before)})
            return {"kept": False, "after": before}
        best_live, best_cost = live(), cost_before
        can_preview = self.max_phase >= 2
        can_persist = self.max_phase >= 3

        if group == 1 and can_preview:
            for d in TRY_DELTAS:
                self._guard_budgets(preview=True)
                self.display.preview(local, char_index, d)
                self.residue[key] += d
                self.previews += 1
                self.display.wait_settled(self.timeout_s)
                got = self.scores(self.shoot(show_frame, f"pv_g{group}m{local}"))[module]
                if _cost(got) < best_cost:
                    best_live, best_cost = live(), _cost(got)
        else:
            # Remote groups: persist-verify-revert (no remote preview in firmware).
            # Below phase 3 this only records the proposal without touching NVS.
            if not can_persist:
                self.deltas.append({"scope": group, "module": local, "charIndex": char_index,
                                    "old": old, "new": old, "proposal": True,
                                    "cost_before": list(cost_before),
                                    "cost_after": list(cost_before)})
                return {"kept": False, "after": before}
            for d in TRY_DELTAS:
                self._guard_budgets()
                base = live()
                self.display.persist(group, kind, base + d, local, max(char_index, 0))
                self.overlay[key] = base + d
                self.persists += 1
                self.display.wait_settled(self.timeout_s)
                got = self.scores(self.shoot(show_frame, f"ps_g{group}m{local}"))[module]
                if _cost(got) < best_cost:
                    best_live, best_cost = base + d, _cost(got)
                else:
                    self.display.persist(group, kind, base, local, max(char_index, 0))
                    self.overlay[key] = base
                    self.persists += 1
                    self.display.wait_settled(self.timeout_s)

        new = best_live
        proposal = group == 1 and new != old and not can_persist
        if group == 1 and new != old and can_persist:
            # Persist winner (previews were volatile; the absolute write
            # also discards any preview residue via reload).
            self._guard_budgets()
            self.display.persist(1, kind, new, local, max(char_index, 0))
            self.persists += 1
            self.display.wait_settled(self.timeout_s)
            self.overlay[key] = new
            self.residue[key] = 0
        elif group != 1:
            # Remote branch persisted (or reverted) every candidate above.
            self.overlay[key] = new
            self.residue[key] = 0
        # Dry-run proposal path: hardware still holds the preview residues;
        # overlay/residue are left untouched so live() stays accurate. The
        # residues are volatile — any reload reverts them.
        after = self.scores(self.shoot(show_frame, f"verify_g{group}m{local}"))[module]
        self.deltas.append({"scope": group, "module": local, "charIndex": char_index,
                            "old": old, "new": new, "proposal": proposal,
                            "cost_before": list(cost_before), "cost_after": list(_cost(after))})
        return {"kept": new != old and not proposal, "after": after}

    def _guard_budgets(self, preview: bool = False):
        if preview and self.previews >= MAX_PREVIEWS:
            raise CalibError("preview budget exhausted")
        if not preview and self.persists >= MAX_PERSISTS:
            raise CalibError("persist budget exhausted")
        if self.sweeps >= MAX_FULL_SWEEPS:
            raise CalibError("sweep budget exhausted")

    def _read_cell(self, group: int, local: int, char_index: int) -> int:
        st = self.display.status()
        if group == 1:
            if char_index < 0:
                mods = st.get("moduleOffsets", [])
                return int(mods[local]) if local < len(mods) else 0
            return 0  # live per-char offsets are not exposed; treat as 0 base
        return 0  # remote live offsets not exposed; persist ratchets relatively

    # -- main ------------------------------------------------------------------
    def run(self) -> dict:
        status = self.display.status()
        contract = self.display.contract()
        if contract.get("contractVersion", 0) != SUPPORTED_CONTRACT:
            raise CalibError(f"unsupported contract {contract.get('contractVersion')}")
        self.total = int(status["totalModules"])
        self.charset = int(status["charset"])
        self.drum = str(status["drumOrder"])
        self.group_widths = self._widths(status)
        self._load_bank()
        snapshot = self.display.snapshot()
        report: dict = {
            "contractVersion": SUPPORTED_CONTRACT,
            "fleet": {"totalModules": self.total, "groupWidths": self.group_widths,
                      "charset": self.charset},
            "result": "needs-human", "reason": "", "deltas": self.deltas,
        }
        try:
            self.display.hold(True)
            self._p0_register()
            if self.max_phase >= 1:
                self._p1_coarse()
            if self.max_phase >= 2:
                self._p2_fine()
            if self.max_phase >= 3:
                self._p3_boundaries()
            self._p4_repeatability()
            ok, reason = self._acceptance()
            report["result"] = "converged" if ok else "needs-human"
            report["reason"] = reason
        except CalibError as exc:
            report["reason"] = str(exc)
            try:
                self.display.restore(snapshot)
            except CalibError as exc2:
                report["reason"] += f" | rollback failed: {exc2}"
        finally:
            try:
                self.display.hold(False)
            except CalibError:
                pass
            report["frames"] = [{k: v for k, v in f.items() if k != "crops"} for f in self.frames]
            report["deltas"] = self.deltas
            report["identity"] = {"checks": self.identity,
                                  "persistent": self.identity_persistent,
                                  "bank": {"source": self.template_source,
                                           "glyphs": sorted(self.templates)}}
            with open(os.path.join(self.photo_dir, "report.json"), "w") as fh:
                json.dump(report, fh, indent=2)
        return report

    def _widths(self, status: dict) -> list[int]:
        total = int(status["totalModules"])
        local = int(status["numModules"])
        groups = int(status.get("groupCount", 1)) or 1
        if groups <= 1:
            return [local]
        # Remote widths unknown to us; assume local width except possibly the
        # last group. The master validates frame length = total either way.
        widths = [local] * groups
        widths[-1] = total - local * (groups - 1)
        return widths

    def _blank(self) -> str:
        return " " * self.total

    # -- phases ------------------------------------------------------------------
    def _p0_register(self):
        self.shoot(self._blank(), "p0_blank")
        self.shoot("H" * self.total, "p0_exposure")
        idx = "".join(chr(ord("A") + (i % 26)) for i in range(self.total))
        rec = self.shoot(idx, "p0_index")
        if len(rec["crops"]) != self.total or any(c.size == 0 for c in rec["crops"]):
            raise CalibError("P0 registration failed: module split mismatch")
        means = [float(c.mean()) for c in rec["crops"]]
        if max(means) - min(means) < 1.0:
            raise CalibError("P0 registration failed: crops indistinguishable")

    def _p1_coarse(self):
        coarse = [" ", "E", "H", "O", "0", "-"]
        bad: set[int] = set()
        votes: dict[int, int] = {}
        for g in coarse:
            rec = self.shoot(g * self.total, f"p1_{ord(g)}")
            flagged: set[int] = set()
            for i, s in enumerate(self.scores(rec)):
                if s["verdict"] != "ok":
                    bad.add(i)
            flagged.update(self.consensus(rec, g))
            flagged.update(self.absolute(rec, g))
            for i in flagged:
                votes[i] = votes.get(i, 0) + 1
            for i, crop in enumerate(rec["crops"]):
                if i not in flagged and float(vision.to_gray(crop).std()) >= vision.IDENTITY_MIN_STD:
                    self.bank_samples.setdefault(g, []).append(crop)
        for module in sorted(bad):
            self._tune_cell(module, -1, "H" * self.total)
        if self._bootstrap_missing():
            self._save_bank("bootstrapped")
        # Wrong-but-clean glyphs: offsets cannot fix a whole-glyph error,
        # so re-verify (one show may have been motion blur) and escalate.
        self._verify_uniform("H", {m for m, v in votes.items() if v >= IDENTITY_P1_VOTES}, "P1")

    def _p2_fine(self):
        n = len(self.drum)
        stride = 6
        suspects: dict[tuple[int, int], str] = {}
        # Per-glyph appearance samples: glyph -> [(module, crop)] gathered
        # across staggered frames (each glyph visits many modules).
        samples: dict[str, list[tuple[int, object]]] = {}
        for k in range(0, n, stride):
            self._guard_budgets()
            frame = "".join(self.drum[(k + i) % n] for i in range(self.total))
            rec = self.shoot(frame, f"p2_stride{k}")
            self.sweeps += stride / n
            for i, s in enumerate(self.scores(rec)):
                if s["verdict"] != "ok":
                    suspects[(i, (k + i) % n)] = frame
                else:
                    samples.setdefault(frame[i], []).append((i, rec["crops"][i]))
                    if float(vision.to_gray(rec["crops"][i]).std()) >= vision.IDENTITY_MIN_STD:
                        self.bank_samples.setdefault(frame[i], []).append(rec["crops"][i])
        for (module, ci), frame in suspects.items():
            self._tune_cell(module, ci, frame)
        self.sweeps = min(MAX_FULL_SWEEPS, self.sweeps + 1)
        if self._bootstrap_missing():
            self._save_bank("bootstrapped")
        # Identity across the sweep: same glyph on different modules must
        # match. Batched re-verify with one uniform show per suspect glyph.
        glyph_suspects: dict[str, set[int]] = {}
        for glyph, items in samples.items():
            modules = sorted({m for m, _ in items})
            if len(modules) < IDENTITY_MIN_MODULES:
                continue
            crops = [c for _, c in items]
            for k in vision.consensus_outliers(crops, self.identity_thresh):
                glyph_suspects.setdefault(glyph, set()).add(modules[k])
        for glyph in sorted(glyph_suspects)[:IDENTITY_MAX_VERIFY_GLYPHS]:
            self._verify_uniform(glyph, glyph_suspects[glyph], "P2")

    def _p3_boundaries(self):
        n = len(self.drum)
        glyph_suspects: dict[str, set[int]] = {}
        for k in range(0, n - 1, 7):
            frame = "".join(self.drum[k + (i % 2)] for i in range(self.total))
            rec = self.shoot(frame, f"p3_pair{k}")
            for i, s in enumerate(self.scores(rec)):
                if s["verdict"] == "double":
                    self._tune_cell(i, (k + (i % 2)) % n, frame)
            for i, crop in enumerate(rec["crops"]):
                expected = frame[i]
                if expected in self.templates and vision.score_crop(crop)["verdict"] == "ok":
                    ranked = vision.identify(crop, self.templates)
                    if not ranked or ranked[0][0] != expected:
                        glyph_suspects.setdefault(expected, set()).add(i)
        for glyph in sorted(glyph_suspects)[:IDENTITY_MAX_VERIFY_GLYPHS]:
            self._verify_uniform(glyph, glyph_suspects[glyph], "P3")

    def _p4_repeatability(self):
        recs = [self.shoot("H" * self.total, f"p4_rep{r}") for r in range(3)]
        for r in range(1, 3):
            for i, (a, b) in enumerate(zip(recs[0]["crops"], recs[r]["crops"])):
                diff = float(abs(a.astype(float) - b.astype(float)).mean())
                if diff > 25.0:
                    raise CalibError(f"P4 unstable: module {i} differs between repeats")

    def _acceptance(self) -> tuple[bool, str]:
        if self.identity_persistent:
            mods = sorted({e["module"] for e in self.identity_persistent})
            return False, f"wrong glyph persists on modules {mods} (see identity log)"
        for glyph in ("E", "H"):
            rec = self.shoot(glyph * self.total, f"accept_{glyph}")
            bad = [(i, s["verdict"]) for i, s in enumerate(self.scores(rec)) if s["verdict"] != "ok"]
            if bad:
                return False, f"acceptance failed on modules {bad}"
            wrong = set(self.consensus(rec, glyph)) | set(self.absolute(rec, glyph))
            if wrong:
                return False, f"acceptance: wrong glyph on modules {sorted(wrong)}"
        return True, "edge glyphs clean, identity verified, repeats stable"


def load_bundled_contract() -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "contract.json")) as fh:
        return json.load(fh)
