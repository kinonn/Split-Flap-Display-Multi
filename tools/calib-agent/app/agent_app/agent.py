"""VLM-driven calibration agent with structural guardrails.

The VLM proposes; the harness disposes. Every display-mutating tool runs
through checks the model cannot talk its way around:

- P0 registration (index strip shown + captured) before any tuning.
- persist needs a prior preview+photo cycle for the same local cell
  (remote groups use persist-verify-revert; the harness auto-captures the
  verify photo and scores after every persist).
- Budgets: steps, previews, persists, full sweeps.
- finish(converged) is validated by the classical acceptance gate
  (seam scores + identity on E/H); failure sends the VLM back to work.

Each tool result bundles classical vision scores WITH the photo, so the
VLM fuses both instead of eyeballing pixels alone.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time

import cv2

from calib import vision
from calib.display import CalibError
from calib.loop import MAX_PERSISTS, MAX_PREVIEWS, SUPPORTED_CONTRACT, Calibrator

MAX_STEPS = 150
KEEP_PHOTOS = 4  # trailing captures kept inline; older ones pruned to text

# UX modes (mirror the deterministic CLI phases): dry-run = read-only
# proposals (Phase 1), preview = +volatile nudges (Phase 2),
# full = +persist and converged verdicts (Phases 3-4).
MODES = ("dry-run", "preview", "full")

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_status",
        "description": "Read display status: geometry, drum order, busy, offsets.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "hold",
        "description": "Engage (true) or release (false) calibration hold on the display.",
        "parameters": {"type": "object", "properties": {
            "active": {"type": "boolean"}},
            "required": ["active"]}}},
    {"type": "function", "function": {
        "name": "show",
        "description": "Show an exact-width frame (length must equal numModules, or "
                       "totalModules for fleet-wide). Returns frameId; does NOT photograph.",
        "parameters": {"type": "object", "properties": {
            "frame": {"type": "string"},
            "tag": {"type": "string", "description": "short label for the photo log"}},
            "required": ["frame", "tag"]}}},
    {"type": "function", "function": {
        "name": "capture",
        "description": "Photograph the display now. Returns per-module seam scores, "
                       "identity outliers vs expected frame, and the photo. "
                       "Pass the frame string currently shown.",
        "parameters": {"type": "object", "properties": {
            "expected": {"type": "string", "description": "frame assumed on display"},
            "tag": {"type": "string"}},
            "required": ["expected", "tag"]}}},
    {"type": "function", "function": {
        "name": "preview",
        "description": "Volatile RAM-only nudge of one local module (no save). "
                       "module 0-based, charIndex -1=coarse else drum index, delta +-32. "
                       "Auto re-photographs and scores afterwards.",
        "parameters": {"type": "object", "properties": {
            "module": {"type": "integer"}, "charIndex": {"type": "integer"},
            "delta": {"type": "integer"}},
            "required": ["module", "charIndex", "delta"]}}},
    {"type": "function", "function": {
        "name": "persist",
        "description": "Save ONE offset cell (scope local or group 2..6 on master; "
                       "kind char|module|display; value absolute). Local cells need "
                       "a prior preview cycle. Auto re-photographs and scores.",
        "parameters": {"type": "object", "properties": {
            "scope": {}, "kind": {"type": "string"}, "value": {"type": "integer"},
            "module": {"type": "integer"}, "charIndex": {"type": "integer"}},
            "required": ["scope", "kind", "value"]}}},
    {"type": "function", "function": {
        "name": "finish",
        "description": "End the run. verdict converged only when acceptance passes "
                       "(validated by the harness, not by you).",
        "parameters": {"type": "object", "properties": {
            "verdict": {"type": "string", "enum": ["converged", "needs-human"]},
            "summary": {"type": "string"}},
            "required": ["verdict", "summary"]}}},
]


def jpeg_bytes(img, max_width: int = 768, quality: int = 65) -> bytes:
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise CalibError("JPEG encode failed")
    return bytes(buf)


def safe_tag(tag: str, fallback: str = "photo") -> str:
    """Filename-safe tag: VLM-supplied tags must never carry path parts."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(tag)).strip("._")
    return (cleaned or fallback)[:80]


def docs_dir() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", ".."))


def load_system_prompt() -> str:
    """Assemble the VLM system prompt from the agent docs (single source)."""
    base = docs_dir()
    prompt_path = os.path.join(base, "PROMPT_SNIPPET.md")
    prod_path = os.path.join(base, "PRODUCTION.md")
    prompt = open(prompt_path).read() if os.path.exists(prompt_path) else ""
    prod = open(prod_path).read() if os.path.exists(prod_path) else ""
    return (
        "You are calibrating a split-flap display through tools. Rules:\n"
        + prompt + "\n\nFull runbook:\n" + prod + "\n\nTool discipline:\n"
        "- Call get_status first; engage hold before any show.\n"
        "- P0 registration: show blank, all-H, then the index strip with a "
        "tag containing 'index', capturing after every show.\n"
        "- One outstanding display op at a time; every show/preview/persist "
        "is followed by capture before judging.\n"
        "- Tune coarse module offsets before per-character ones; one cell "
        "at a time; small deltas first.\n"
        "- Each tool result already contains classical vision scores AND a "
        "photo: fuse both. If they disagree, trust the photo and say so.\n"
        "- Budgets are enforced by the harness; a rejected tool call means "
        "change strategy, not retry identically.\n"
        "- End with finish(converged) only after clean acceptance frames; "
        "otherwise finish(needs-human) with module numbers and evidence.\n"
        "- Mode restrictions for this run are enforced by the harness: "
        "dry-run allows show/capture only (propose offsets in text), "
        "preview additionally allows volatile preview nudges, and only "
        "full allows persist and finish(converged)."
    )


class Agent:
    """ReAct loop over a VLM client, a Calibrator (display+camera owner) and
    an event sink (UI log). The vlm object needs .chat(messages, tools)."""

    def __init__(self, vlm, calib: Calibrator, system_prompt: str,
                 on_event=None, max_steps: int = MAX_STEPS,
                 mode: str = "full"):
        self.vlm = vlm
        self.calib = calib
        self.system_prompt = system_prompt
        self.on_event = on_event or (lambda e: None)
        self.max_steps = max_steps
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.steps = 0
        self.p0_done = False
        self.last_frame = ""
        self.previewed: set[tuple] = set()
        self.aborted = False
        self.report: dict | None = None

    def event(self, kind: str, text: str, photo: str | None = None,
              detail: dict | None = None):
        evt: dict = {"t": time.strftime("%H:%M:%S"), "kind": kind,
                      "text": text, "photo": photo}
        if detail is not None:
            evt["detail"] = detail
        self.on_event(evt)

    # -- tool implementations -------------------------------------------------
    def _photo_record(self, frame: str, tag: str) -> dict:
        img = self.calib.camera.capture()
        path = os.path.join(self.calib.photo_dir, f"{safe_tag(tag)}.png")
        if not cv2.imwrite(path, img):
            raise CalibError(f"failed to write photo {os.path.basename(path)}")
        gray = vision.to_gray(img)
        crops = vision.split_crops(gray, self.calib.total)
        scores = [vision.score_crop(c) for c in crops]
        return {"photo": os.path.basename(path), "image": jpeg_bytes(img),
                "scores": [{"module": i, **s} for i, s in enumerate(scores)],
                "crops": crops}

    def tool_get_status(self, args: dict) -> dict:
        st = self.calib.display.status()
        return {"status": {k: v for k, v in st.items() if k != "moduleOffsets"},
                "moduleOffsets": st.get("moduleOffsets")}

    def tool_hold(self, args: dict) -> dict:
        return self.calib.display.hold(bool(args["active"]))

    def tool_show(self, args: dict) -> dict:
        frame, tag = args["frame"], args.get("tag", f"step{self.steps}")
        info = self.calib.display.show_and_settle(frame, self.calib.dwell_ms,
                                                  self.calib.timeout_s)
        self.last_frame = frame
        self.event("show", f"show {tag!r} frameId={info['frameId']}",
                   detail={"frame": frame, "frameId": info["frameId"],
                           "fleetFrame": info.get("fleetFrame", False)})
        if "index" in tag:
            self.p0_done = True
        return info

    def tool_capture(self, args: dict) -> dict:
        expected, tag = args["expected"], args.get("tag", f"cap{self.steps}")
        rec = self._photo_record(expected, f"{tag}")
        matched = len(expected) == len(rec["crops"])
        outliers = self._identity_on_crops(rec["crops"], expected, tag) if matched else []
        if matched:
            # Feed the template bank with trusted crops (clean + agreeing),
            # exactly like the deterministic runner's bootstrap. Golden
            # banks are never auto-extended (see Calibrator).
            for i, crop in enumerate(rec["crops"]):
                if i not in outliers and rec["scores"][i]["verdict"] == "ok" \
                        and float(vision.to_gray(crop).std()) >= vision.IDENTITY_MIN_STD:
                    self.calib.bank_samples.setdefault(expected[i], []).append(crop)
            if self.calib._bootstrap_missing():
                self.calib._save_bank("bootstrapped")
        self.event("photo", f"{tag}: " + ", ".join(
            f"m{s['module']}={s['verdict']}" for s in rec["scores"]), rec["photo"],
            detail={"frame": expected, "scores": rec["scores"],
                    "identity_outliers": outliers})
        out = {"photo": rec["photo"], "scores": rec["scores"], "_image": rec["image"]}
        if len(expected) == len(rec["crops"]):
            out["identity_outliers"] = outliers
        return out

    def _identity_on_crops(self, crops, expected: str, tag: str = "") -> list[int]:
        ok_idx = [i for i, c in enumerate(crops)
                  if vision.score_crop(c)["verdict"] == "ok"]
        out: set[int] = set()
        if len(ok_idx) >= 3:
            sub = [crops[i] for i in ok_idx]
            out.update(ok_idx[k] for k in
                       vision.consensus_outliers(sub, self.calib.identity_thresh))
        for i in ok_idx:
            if expected[i] in self.calib.templates:
                ranked = vision.identify(crops[i], self.calib.templates)
                if (not ranked or ranked[0][0] != expected[i]
                        or ranked[0][1] < vision.IDENTITY_ABSOLUTE_MIN):
                    out.add(i)
        self.calib.identity.append({"tag": tag, "glyph": expected,
                                    "method": "consensus+template",
                                    "outliers": sorted(out)})
        return sorted(out)

    def tool_preview(self, args: dict) -> dict:
        if self.mode == "dry-run":
            raise CalibError("dry-run mode: previews disabled, show+capture only")
        if not self.p0_done:
            raise CalibError("P0 registration first: show+capture the index strip")
        module, ci, delta = int(args["module"]), int(args["charIndex"]), int(args["delta"])
        n = self.calib.display.status()["numModules"]
        if not (0 <= module < n):
            raise CalibError(f"module out of range 0..{n - 1}")
        if self.calib.previews >= MAX_PREVIEWS:
            raise CalibError("preview budget exhausted")
        self.calib.display.preview(module, ci, delta)
        self.calib.previews += 1
        self.calib.display.wait_settled(self.calib.timeout_s)
        group = 1
        self.previewed.add((group, module, ci))
        # Structural verify-after: photo + scores are part of the result.
        frame = self.last_frame or " " * self.calib.total
        rec = self._photo_record(frame, f"pv_m{module}c{ci}")
        self.event("preview", f"preview m{module} c{ci} {delta:+d}", rec["photo"],
                   detail={"frame": frame, "scores": rec["scores"],
                           "module": module, "charIndex": ci, "delta": delta})
        return {"scores": rec["scores"], "photo": rec["photo"], "_image": rec["image"]}

    def tool_persist(self, args: dict) -> dict:
        if self.mode in ("dry-run", "preview"):
            raise CalibError(f"{self.mode} mode: persisting disabled, "
                             "propose values in text instead")
        if not self.p0_done:
            raise CalibError("P0 registration first")
        scope, kind = args["scope"], args["kind"]
        group = 1 if scope in ("local", 1, "1") else int(scope)
        module, ci = int(args.get("module", 0)), int(args.get("charIndex", 0))
        if kind == "module":
            ci = -1  # coarse cell: charIndex is meaningless, normalize it
        if group == 1 and kind in ("char", "module") and (group, module, ci) not in self.previewed:
            raise CalibError(f"preview module {module} charIndex {ci} first (with photo)")
        if self.calib.persists >= MAX_PERSISTS:
            raise CalibError("persist budget exhausted")
        if kind == "char":
            self.calib.overlay[(group, module, ci)] = int(args["value"])
        resp = self.calib.display.persist(scope, kind, int(args["value"]), module, ci)
        self.calib.persists += 1
        self.calib.display.wait_settled(self.calib.timeout_s)
        frame = self.last_frame or " " * self.calib.total
        rec = self._photo_record(frame, f"ps_g{group}m{module}")
        self.event("persist", f"persist g{group} {kind}={args['value']}", rec["photo"],
                   detail={"frame": frame, "scores": rec["scores"], "scope": group,
                           "kind": kind, "module": module,
                           "charIndex": ci, "value": int(args["value"])})
        return {"saved": resp, "scores": rec["scores"],
                "photo": rec["photo"], "_image": rec["image"]}

    def tool_finish(self, args: dict) -> dict:
        verdict = args["verdict"]
        if verdict == "converged" and self.mode != "full":
            return {"accepted": False,
                    "reason": f"{self.mode} mode cannot converge: end with "
                              "finish(needs-human) and your proposal summary."}
        if verdict == "converged":
            ok, reason = self.calib._acceptance()
            if not ok:
                return {"accepted": False,
                        "reason": f"harness acceptance failed: {reason}. Keep working."}
            return {"accepted": True, "reason": reason}
        return {"accepted": True, "reason": "needs-human accepted"}

    # -- loop -------------------------------------------------------------------
    TOOLS = {"get_status": tool_get_status, "hold": tool_hold, "show": tool_show,
             "capture": tool_capture, "preview": tool_preview,
             "persist": tool_persist, "finish": tool_finish}

    def run(self) -> dict:
        calib = self.calib
        try:
            status = calib.display.status()
        except CalibError as exc:
            return self._done("needs-human", f"display unreachable: {exc}")
        if status.get("contractVersion", 0) != SUPPORTED_CONTRACT:
            return self._done("needs-human", "unsupported contract version")
        calib.total = int(status["totalModules"])
        calib.charset = int(status["charset"])
        calib.drum = str(status["drumOrder"])
        calib.group_widths = calib._widths(status)
        calib._load_bank()
        snapshot = calib.display.snapshot()
        with open(os.path.join(calib.photo_dir, "snapshot.json"), "w") as fh:
            json.dump(snapshot, fh)
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content":
                     f"Begin calibration in {self.mode!r} mode. Fleet: {calib.total} modules, "
                     f"charset {calib.charset}. Snapshot taken. Engage hold on the display, "
                     f"then run P0 registration (blank, all-H, index strip)."}]
        self.event("run", f"started: {calib.total} modules, charset {calib.charset}")
        while self.steps < self.max_steps and not self.aborted:
            self.steps += 1
            try:
                reply = self.vlm.chat(messages, TOOL_SCHEMAS)
            except Exception as exc:
                messages.append({"role": "assistant", "content": f"VLM error: {exc}"})
                if self.steps > 5:
                    return self._done("needs-human", f"VLM failing repeatedly: {exc}")
                continue
            assistant: dict = {"role": "assistant"}
            if reply.get("content"):
                assistant["content"] = reply["content"]
            if reply.get("tool_calls"):
                assistant["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"],
                                 "arguments": json.dumps(c["arguments"])}} for c in reply["tool_calls"]]
            messages.append(assistant)
            if not reply.get("tool_calls"):
                self.event("vlm", (reply.get("content") or "")[:500])
                continue
            done = None  # the accepted finish call, if any
            for call in reply["tool_calls"]:
                result = self._execute(call["name"], call["arguments"])
                images = [result.pop("_image", None)]
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(result, default=str)[:4000]})
                if images[0] is not None:
                    messages.append({"role": "user", "content": [
                        {"type": "text", "text": f"Photo for {call['name']} "
                         f"(frame context in tool result above)."},
                        {"type": "image_url", "image_url": {
                            "url": "data:image/jpeg;base64," + base64.b64encode(images[0]).decode()}}]})
                if call["name"] == "finish" and result.get("accepted"):
                    # Stop here: no display ops may run after the verdict,
                    # and summary/reason must come from THIS call.
                    done = call
                    break
            self._prune(messages)
            if done:
                verdict = done["arguments"].get("verdict", "needs-human")
                return self._done(verdict, f"{done['arguments'].get('summary', '')} | "
                                           f"{result.get('reason', '')}")
        reason = "aborted by user" if self.aborted else "step budget exhausted"
        return self._done("needs-human", reason)

    def _execute(self, name: str, args: dict) -> dict:
        fn = self.TOOLS.get(name)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            out = fn(self, args)
            return out if isinstance(out, dict) else {"result": out}
        except CalibError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # never let one tool kill the run
            return {"error": f"harness error: {exc}"}

    def _prune(self, messages: list):
        """Keep only trailing captures inline; older photos become text."""
        seen = 0
        for msg in reversed(messages):
            content = msg.get("content")
            if isinstance(content, list):
                if seen >= KEEP_PHOTOS:
                    msg["content"] = [{"type": "text", "text": "[earlier photo omitted]"}]
                else:
                    seen += 1

    def _done(self, verdict: str, reason: str) -> dict:
        try:
            self.calib.display.hold(False)
        except Exception:
            pass
        self.report = {"result": verdict, "reason": reason, "steps": self.steps,
                       "identity": {"checks": self.calib.identity,
                                    "persistent": self.calib.identity_persistent,
                                    "bank": {"source": self.calib.template_source,
                                             "glyphs": sorted(self.calib.templates)}}}
        path = os.path.join(self.calib.photo_dir, "agent_report.json")
        with open(path, "w") as fh:
            json.dump(self.report, fh, indent=2)
        self.event("done", f"{verdict}: {reason}")
        return self.report
