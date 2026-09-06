"""Tests for the agent guardrails with faked display/camera/VLM."""

import numpy as np

from agent_app.agent import TOOL_SCHEMAS, Agent
from calib import vision
from calib.loop import Calibrator


class FakeDisplay:
    def __init__(self, total=4):
        self.total = total
        self.frames = 0
        self.previews = []
        self.persists = []
        self.held = False

    def status(self):
        return {"totalModules": self.total, "numModules": self.total,
                "groupCount": 1, "charset": 37,
                "drumOrder": " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                "contractVersion": 1, "mode": 4, "moduleOffsets": [0] * self.total}

    def contract(self):
        return {"contractVersion": 1}

    def hold(self, active):
        self.held = bool(active)
        return {"holdActive": self.held}

    def snapshot(self):
        return {"settings": {}}

    def show_and_settle(self, frame, dwell_ms=800, timeout_s=60):
        self.frames += 1
        return {"frameId": self.frames, "fleetFrame": False}

    def wait_settled(self, timeout_s=60):
        return self.status()

    def preview(self, module, char_index, delta):
        self.previews.append((module, char_index, delta))
        return {"message": "queued"}

    def persist(self, scope, kind, value, module=0, char_index=0):
        self.persists.append((scope, kind, value))
        return {"message": "saved"}


class FakeCamera:
    def __init__(self, total=4):
        self.total = total

    def capture(self):
        h, w = 96, 68 * self.total
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(self.total):
            frame[:, i * 68:(i + 1) * 68] = 150 + i * 10
        return frame


class ScriptVLM:
    """Replies with a scripted list of (content, [(tool, args)]) turns."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools):
        assert tools, "harness must always offer tools"
        content, calls = self.script.pop(0)
        self.calls.append(calls)
        return {"content": content,
                "tool_calls": [{"id": f"c{i}", "name": n, "arguments": a}
                               for i, (n, a) in enumerate(calls)]}


def _agent(script, tmp_path):
    display, camera = FakeDisplay(), FakeCamera()
    calib = Calibrator(display, camera, photo_dir=str(tmp_path),
                       dwell_ms=0, timeout_s=5)
    agent = Agent(ScriptVLM(script), calib, "system", on_event=lambda e: None)
    return agent, display


def test_persist_without_preview_rejected(tmp_path):
    script = [
        (None, [("hold", {"active": True})]),
        (None, [("show", {"frame": "ABCD", "tag": "p0_index"})]),
        (None, [("capture", {"expected": "ABCD", "tag": "c"})]),
        (None, [("persist", {"scope": "local", "kind": "module",
                             "module": 0, "value": 5})]),
        ("stuck", []),
        (None, [("finish", {"verdict": "needs-human", "summary": "blocked"})]),
    ]
    agent, display = _agent(script, tmp_path)
    report = agent.run()
    assert display.persists == []
    assert report["result"] == "needs-human"


def test_happy_path_persists_after_preview(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "split_crops",
                        lambda gray, n: [gray[:, i * 68:(i + 1) * 68] for i in range(n)])
    script = [
        (None, [("hold", {"active": True})]),
        (None, [("show", {"frame": "HHHH", "tag": "p0_index"})]),
        (None, [("capture", {"expected": "HHHH", "tag": "c"})]),
        (None, [("preview", {"module": 0, "charIndex": -1, "delta": 2})]),
        (None, [("persist", {"scope": "local", "kind": "module",
                             "module": 0, "charIndex": -1, "value": 2})]),
        (None, [("finish", {"verdict": "needs-human", "summary": "demo over"})]),
    ]
    agent, display = _agent(script, tmp_path)
    report = agent.run()
    assert len(display.previews) == 1
    assert len(display.persists) == 1
    assert report["result"] == "needs-human"
    assert display.held is False  # hold released at end


def test_tool_schemas_cover_all_tools():
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == set(Agent.TOOLS)


def _agent_with_mode(script, tmp_path, mode, monkeypatch=None):
    display, camera = FakeDisplay(), FakeCamera()
    calib = Calibrator(display, camera, photo_dir=str(tmp_path),
                       dwell_ms=0, timeout_s=5)
    agent = Agent(ScriptVLM(script), calib, "system", on_event=lambda e: None,
                  mode=mode)
    return agent, display


def test_invalid_mode_rejected(tmp_path):
    import pytest

    display, camera = FakeDisplay(), FakeCamera()
    calib = Calibrator(display, camera, photo_dir=str(tmp_path),
                       dwell_ms=0, timeout_s=5)
    with pytest.raises(ValueError):
        Agent(ScriptVLM([]), calib, "system", mode="turbo")


def test_dry_run_blocks_preview_persist_and_converge(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "split_crops",
                        lambda gray, n: [gray[:, i * 68:(i + 1) * 68] for i in range(n)])
    script = [
        (None, [("show", {"frame": "HHHH", "tag": "p0_index"})]),
        (None, [("capture", {"expected": "HHHH", "tag": "c"})]),
        (None, [("preview", {"module": 0, "charIndex": -1, "delta": 2})]),
        (None, [("persist", {"scope": "local", "kind": "module",
                             "module": 0, "value": 2})]),
        (None, [("finish", {"verdict": "converged", "summary": "done?"})]),
        (None, [("finish", {"verdict": "needs-human", "summary": "dry run over"})]),
    ]
    agent, display = _agent_with_mode(script, tmp_path, "dry-run")
    report = agent.run()
    assert display.previews == []
    assert display.persists == []
    assert report["result"] == "needs-human"
    assert report["reason"].endswith("dry run over | needs-human accepted")


def test_preview_mode_allows_preview_blocks_persist(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "split_crops",
                        lambda gray, n: [gray[:, i * 68:(i + 1) * 68] for i in range(n)])
    script = [
        (None, [("show", {"frame": "HHHH", "tag": "p0_index"})]),
        (None, [("capture", {"expected": "HHHH", "tag": "c"})]),
        (None, [("preview", {"module": 1, "charIndex": -1, "delta": -2})]),
        (None, [("persist", {"scope": "local", "kind": "module",
                             "module": 1, "value": -2})]),
        (None, [("finish", {"verdict": "needs-human", "summary": "preview over"})]),
    ]
    agent, display = _agent_with_mode(script, tmp_path, "preview")
    report = agent.run()
    assert len(display.previews) == 1
    assert display.persists == []
    assert report["result"] == "needs-human"


def test_capture_event_carries_transparency_detail(tmp_path, monkeypatch):
    from agent_app.agent import Agent as AgentCls

    monkeypatch.setattr(vision, "split_crops",
                        lambda gray, n: [gray[:, i * 68:(i + 1) * 68] for i in range(n)])
    events = []
    display, camera = FakeDisplay(), FakeCamera()
    calib = Calibrator(display, camera, photo_dir=str(tmp_path),
                       dwell_ms=0, timeout_s=5)
    agent = AgentCls(ScriptVLM([
        (None, [("show", {"frame": "HHHH", "tag": "t"})]),
        (None, [("capture", {"expected": "HHHH", "tag": "c"})]),
        (None, [("finish", {"verdict": "needs-human", "summary": "x"})]),
    ]), calib, "system", on_event=events.append)
    agent.run()
    photos = [e for e in events if e["kind"] == "photo"]
    assert len(photos) == 1
    detail = photos[0]["detail"]
    assert detail["frame"] == "HHHH"
    assert [s["module"] for s in detail["scores"]] == [0, 1, 2, 3]
    assert all("verdict" in s for s in detail["scores"])
    assert "identity_outliers" in detail


def test_slash_in_capture_tag_still_writes_photo(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "split_crops",
                        lambda gray, n: [gray[:, i * 68:(i + 1) * 68] for i in range(n)])
    script = [
        (None, [("show", {"frame": "HHHH", "tag": "p0_index"})]),
        (None, [("capture", {"expected": "HHHH", "tag": "sub/dir/cap"})]),
        (None, [("finish", {"verdict": "needs-human", "summary": "x"})]),
    ]
    agent, display = _agent(script, tmp_path)
    agent.run()
    assert (tmp_path / "sub_dir_cap.png").is_file()
    assert not (tmp_path / "sub").exists()


def test_finish_summary_kept_and_no_ops_after_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "split_crops",
                        lambda gray, n: [gray[:, i * 68:(i + 1) * 68] for i in range(n)])
    script = [
        (None, [("show", {"frame": "HHHH", "tag": "p0_index"})]),
        (None, [("capture", {"expected": "HHHH", "tag": "c"})]),
        (None, [("preview", {"module": 0, "charIndex": -1, "delta": 2})]),
        (None, [("finish", {"verdict": "needs-human", "summary": "stop here"}),
                ("persist", {"scope": "local", "kind": "module",
                             "module": 0, "value": 9})]),
    ]
    agent, display = _agent(script, tmp_path)
    report = agent.run()
    assert report["result"] == "needs-human"
    # summary + reason come from the finish call, not a later tool result
    assert "stop here" in report["reason"]
    assert "needs-human accepted" in report["reason"]
    assert display.persists == []  # nothing may run after the verdict


def test_safe_tag_never_carries_path_parts():
    from agent_app.agent import safe_tag

    assert safe_tag("sub/dir/cap") == "sub_dir_cap"
    assert safe_tag("..", "fb") == "fb"
    assert len(safe_tag("x" * 300)) == 80
