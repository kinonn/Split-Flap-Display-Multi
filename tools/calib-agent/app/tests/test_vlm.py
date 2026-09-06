"""Tests for agent_app/vlm.py with a faked transport."""

import json

import pytest

from agent_app import vlm


class FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


def test_chat_parses_tool_calls(monkeypatch):
    calls = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        calls["url"] = url
        calls["json"] = json
        return FakeResp({"choices": [{"message": {
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "hold",
                                         "arguments": '{"active": true}'}}]}}]})

    monkeypatch.setattr(vlm.requests, "post", fake_post)
    client = vlm.VLMClient("https://llm.example/v1", "m", "k")
    out = client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
    assert calls["url"] == "https://llm.example/v1/chat/completions"
    assert calls["json"]["model"] == "m"
    assert out["tool_calls"] == [{"id": "c1", "name": "hold", "arguments": {"active": True}}]


def test_chat_text_reply(monkeypatch):
    monkeypatch.setattr(vlm.requests, "post",
                        lambda *a, **k: FakeResp({"choices": [{"message": {
                            "content": "done", "tool_calls": []}}]}))
    out = vlm.VLMClient("https://x", "m", "k").chat([{"role": "user", "content": "hi"}])
    assert out == {"content": "done", "tool_calls": []}


def test_http_error_maps_to_vlm_error(monkeypatch):
    monkeypatch.setattr(vlm.requests, "post",
                        lambda *a, **k: FakeResp({"error": "nope"}, status=500))
    with pytest.raises(vlm.VLMError):
        vlm.VLMClient("https://x", "m", "k").chat([])


def test_image_part_is_data_url():
    part = vlm.image_part(b"\xff\xd8fake")
    assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_opencode_headers_only_for_opencode(monkeypatch):
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(headers or {})
        return FakeResp({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(vlm.requests, "post", fake_post)
    vlm.VLMClient("https://opencode.ai/zen/go/v1", "deepseek-v4-flash", "k",
                  session_id="abc").chat([])
    assert seen["X-Opencode-Session"] == "splitflap-calib-abc"
    assert seen["User-Agent"].startswith("splitflap-calib-agent/")

    seen.clear()
    vlm.VLMClient("https://api.openai.com/v1", "gpt-4o", "k").chat([])
    assert "X-Opencode-Session" not in seen
