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


def _direct_agent(tmp_path):
    """Agent without running the loop (direct tool calls)."""
    display, camera = FakeDisplay(), FakeCamera()
    calib = Calibrator(display, camera, photo_dir=str(tmp_path),
                       dwell_ms=0, timeout_s=5)
    calib.total = 4
    agent = Agent(ScriptVLM([]), calib, "system", on_event=lambda e: None)
    return agent, display, calib


def test_p0_show_without_capture_blocks_preview(tmp_path):
    # Issue kinonn-bot#29: the tag alone must not complete P0.
    script = [
        (None, [("hold", {"active": True})]),
        (None, [("show", {"frame": "ABCD", "tag": "p0_index"})]),
        (None, [("preview", {"module": 0, "charIndex": -1, "delta": 2})]),
        (None, [("finish", {"verdict": "needs-human", "summary": "no p0"})]),
    ]
    agent, display = _agent(script, tmp_path)
    report = agent.run()
    assert agent.p0_done is False
    assert display.previews == []
    assert report["result"] == "needs-human"


def test_p0_mismatched_capture_verifies_nothing(tmp_path):
    # Issue kinonn-bot#29/#33: wrong expected frame proves nothing.
    script = [
        (None, [("hold", {"active": True})]),
        (None, [("show", {"frame": "ABCD", "tag": "p0_index"})]),
        (None, [("capture", {"expected": "WXYZ", "tag": "c"})]),
        (None, [("finish", {"verdict": "needs-human", "summary": "mismatch"})]),
    ]
    agent, display = _agent(script, tmp_path)
    report = agent.run()
    assert agent.p0_done is False
    assert agent.calib.templates == {}
    assert report["result"] == "needs-human"


def test_hold_release_rejected_mid_run(tmp_path):
    # Issue kinonn-bot#31: hold is harness-owned.
    import pytest

    from calib.display import CalibError

    agent, display, _ = _direct_agent(tmp_path)
    assert agent.tool_hold({"active": True})["holdActive"] is True
    with pytest.raises(CalibError, match="harness-owned"):
        agent.tool_hold({"active": False})


def test_remote_and_display_persist_need_prior_capture(tmp_path):
    # Issue kinonn-bot#30: no blind writes off the local preview path.
    import pytest

    from calib.display import CalibError

    agent, display, _ = _direct_agent(tmp_path)
    agent.p0_done = True
    with pytest.raises(CalibError, match="capture the display first"):
        agent.tool_persist({"scope": 2, "kind": "module",
                            "module": 0, "value": 7})
    with pytest.raises(CalibError, match="capture the display first"):
        agent.tool_persist({"scope": "local", "kind": "display",
                            "value": 3})
    assert display.persists == []


def test_remote_persist_allowed_after_p0_capture(tmp_path):
    script = [
        (None, [("hold", {"active": True})]),
        (None, [("show", {"frame": "ABCD", "tag": "p0_index"})]),
        (None, [("capture", {"expected": "ABCD", "tag": "c"})]),
        (None, [("persist", {"scope": 2, "kind": "module",
                             "module": 0, "value": 7})]),
        (None, [("finish", {"verdict": "needs-human", "summary": "remote ok"})]),
    ]
    agent, display = _agent(script, tmp_path)
    report = agent.run()
    assert agent.p0_done is True
    assert len(display.persists) == 1
    assert report["result"] == "needs-human"


def test_finish_converged_needs_p0_and_bank(tmp_path):
    # Issue kinonn-bot#32: converged by omission is not converged.
    agent, _, _ = _direct_agent(tmp_path)
    out = agent.tool_finish({"verdict": "converged", "summary": "x"})
    assert out["accepted"] is False
    assert "P0" in out["reason"]
    agent.p0_done = True
    out = agent.tool_finish({"verdict": "converged", "summary": "x"})
    assert out["accepted"] is False
    assert "template bank" in out["reason"]


def test_blind_camera_fails_image_sanity(tmp_path):
    # Issue kinonn-bot#32: flat frames carry no glyph signal.
    agent, _, _ = _direct_agent(tmp_path)
    sane, _ = agent._acceptance_image_sanity()
    assert sane is False


def test_mismatched_capture_warns_and_skips_bank(tmp_path, monkeypatch):
    # Issue kinonn-bot#33: only verified captures feed P0/bank/identity.
    def distinct_crops(gray, n):
        crops = []
        for i in range(n):
            c = np.zeros((96, 68), dtype=np.uint8)
            c[:, 34:] = 255
            crops.append(np.clip(c.astype(int) + i * 10, 0, 255).astype(np.uint8))
        return crops

    monkeypatch.setattr(vision, "split_crops", distinct_crops)
    agent, _, calib = _direct_agent(tmp_path)
    agent.tool_show({"frame": "ABCD", "tag": "p0_index"})
    out = agent.tool_capture({"expected": "WXYZ", "tag": "c"})
    assert out["verified"] is False
    assert "warning" in out
    assert agent.p0_done is False
    assert calib.bank_samples == {}
    out = agent.tool_capture({"expected": "ABCD", "tag": "c2"})
    assert out["verified"] is True
    assert agent.p0_done is True


def test_abort_during_model_call_returns_promptly(tmp_path):
    # Issue kinonn-bot#36: a hung gateway call must not delay abort.
    import threading
    import time

    from agent_app.agent import Agent as AgentCls
    from calib.loop import Calibrator as CalibratorCls

    class SlowVLM:
        def chat(self, messages, tools):
            time.sleep(30)  # hung provider; abort must cut this short
            return {"content": "late", "tool_calls": []}

    display, camera = FakeDisplay(), FakeCamera()
    calib = CalibratorCls(display, camera, photo_dir=str(tmp_path),
                          dwell_ms=0, timeout_s=5)
    agent = AgentCls(SlowVLM(), calib, "system", on_event=lambda e: None)
    box = {}
    t = threading.Thread(target=lambda: box.update(report=agent.run()))
    t0 = time.monotonic()
    t.start()
    time.sleep(2.0)  # let the run reach the blocking model call
    agent.aborted = True
    t.join(timeout=15)
    dt = time.monotonic() - t0
    assert not t.is_alive(), "run ignored abort during the model call"
    assert dt < 15, f"abort took {dt:.1f}s"
    assert box["report"]["result"] == "needs-human"
    assert box["report"]["reason"] == "aborted by user"


def test_malformed_tool_arguments_surface_explicitly(tmp_path):
    # Issue kinonn-bot#36: non-JSON arguments must name the real problem.
    agent, _, _ = _direct_agent(tmp_path)
    out = agent._execute("show", {"_parse_error": "{oops"})
    assert "not valid JSON" in out["error"]
    assert "{oops" in out["error"]


def test_agent_preview_uses_calibrator_instance_budget(tmp_path):
    # full_drum scales the Calibrator caps; the agent must honor the
    # instance caps, not the module-level defaults.
    from calib.loop import MAX_PREVIEWS, Calibrator

    agent, _, _ = _direct_agent(tmp_path)
    agent.p0_done = True
    args = {"module": 0, "charIndex": -1, "delta": 2}
    agent.calib.previews = MAX_PREVIEWS
    out = agent._execute("preview", args)
    assert out["error"] == "preview budget exhausted"
    full_calib = Calibrator(agent.calib.display, agent.calib.camera,
                            photo_dir=str(tmp_path), dwell_ms=0, timeout_s=5,
                            full=True)
    full_calib.total = agent.calib.total  # run() normally sets this
    agent.calib = full_calib
    agent.calib.previews = MAX_PREVIEWS  # old cap: still headroom when full
    out = agent._execute("preview", args)
    assert "error" not in out, out


def test_multi_tool_turn_keeps_tool_responses_adjacent(tmp_path, monkeypatch):
    # Strict providers 400 the whole history when a user message (the
    # capture photo) lands between two tool responses of one assistant
    # tool_calls block ("must be followed by tool messages responding to
    # each tool_call_id"). Photos must be buffered until after ALL
    # responses. Issue: run died at step 33 with exactly that 400.
    monkeypatch.setattr(vision, "split_crops",
                        lambda gray, n: [gray[:, i * 68:(i + 1) * 68] for i in range(n)])
    script = [
        (None, [("show", {"frame": "HHHH", "tag": "p0_index"}),
                ("capture", {"expected": "HHHH", "tag": "c"})]),
        (None, [("finish", {"verdict": "needs-human", "summary": "order ok"})]),
    ]
    agent, _ = _agent(script, tmp_path)
    report = agent.run()
    assert report["result"] == "needs-human"
    # Capture the message order via a recording VLM wrapper.
    recorded = []

    class RecordingVLM(ScriptVLM):
        def chat(self, messages, tools):
            recorded.append([dict(m, content=m.get("content")) for m in messages])
            return super().chat(messages, tools)

    display, camera = FakeDisplay(), FakeCamera()
    calib = Calibrator(display, camera, photo_dir=str(tmp_path),
                       dwell_ms=0, timeout_s=5)
    agent2 = Agent(RecordingVLM([
        (None, [("show", {"frame": "HHHH", "tag": "p0_index"}),
                ("capture", {"expected": "HHHH", "tag": "c"})]),
        (None, [("finish", {"verdict": "needs-human", "summary": "order ok"})]),
    ]), calib, "system", on_event=lambda e: None)
    agent2.run()
    final = recorded[-1]
    # Find the assistant message with two tool_calls and verify both tool
    # responses are adjacent, with photos only after them.
    for i, m in enumerate(final):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            ids = [c["id"] for c in m["tool_calls"]]
            assert len(ids) == 2
            roles = [final[j].get("role") for j in range(i + 1, i + 3)]
            assert roles == ["tool", "tool"], roles
            answered = [final[j].get("tool_call_id") for j in range(i + 1, i + 3)]
            assert answered == ids
            # Any photo user messages come after both tool responses.
            for j in range(i + 3, len(final)):
                if final[j].get("role") == "assistant":
                    break
                assert final[j].get("role") in ("tool", "user")


def test_repair_tool_messages_fills_orphans():
    from agent_app.agent import Agent

    messages = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "show", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "capture", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "{}"},
        {"role": "user", "content": "photo"},
    ]
    fixed = Agent._repair_tool_messages(messages)
    assert fixed == 1
    roles = [m["role"] for m in messages]
    assert roles == ["system", "assistant", "tool", "tool", "user"]
    assert messages[2]["tool_call_id"] == "a"
    assert messages[3]["tool_call_id"] == "b"
    # Idempotent: a second pass inserts nothing.
    assert Agent._repair_tool_messages(messages) == 0


def test_vlm_synthesizes_missing_tool_call_ids(monkeypatch):
    from agent_app import vlm as vlm_mod

    class FakeResp:
        def __init__(self, payload):
            self._payload = payload
            self.status_code = 200
            self.text = "{}"

        def json(self):
            return self._payload

    def fake_post(self, url, json=None, headers=None, timeout=None):
        return FakeResp({"choices": [{"message": {
            "content": None,
            "tool_calls": [
                {"id": None, "type": "function",
                 "function": {"name": "show", "arguments": "{}"}},
                {"type": "function",
                 "function": {"name": "capture", "arguments": "{}"}},
            ]}}]})

    monkeypatch.setattr(vlm_mod.requests.Session, "post", fake_post)
    out = vlm_mod.VLMClient("https://x", "m", "k").chat([])
    ids = [c["id"] for c in out["tool_calls"]]
    assert ids == ["call_0", "call_1"]
    assert all(ids)
