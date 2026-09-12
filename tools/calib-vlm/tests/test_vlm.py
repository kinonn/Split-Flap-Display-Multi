"""VLMClient tests: forced tool_choice fallback for thinking providers."""

import pytest

import calib_vlm.vlm as vlm


class FakeResp:
    def __init__(self, status_code=200, text="", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_forced_tool_choice_rejected_by_thinking_provider(monkeypatch):
    seen = []

    def fake_post(self, url, json=None, headers=None, timeout=None):
        seen.append(dict(json))
        if "tool_choice" in json:
            return FakeResp(
                400, '{"error":{"message":"Thinking mode does not '
                     'support this tool_choice"}}')
        return FakeResp(200, payload={"choices": [{"message": {
            "content": "ok", "tool_calls": []}}]})

    monkeypatch.setattr(vlm.requests.Session, "post", fake_post)
    client = vlm.VLMClient("https://x", "m", "k")
    out = client.chat([{"role": "user", "content": "hi"}],
                      tools=[{"type": "function",
                              "function": {"name": "t"}}],
                      tool_choice="required")
    assert out == {"content": "ok", "tool_calls": []}
    # First attempt forced the choice, the retry dropped it.
    assert seen[0]["tool_choice"] == "required"
    assert "tool_choice" not in seen[1]
    # Sticky: later calls skip the forced choice (no wasted round trip).
    seen.clear()
    client.chat([], tools=[{"type": "function",
                            "function": {"name": "t"}}])
    assert len(seen) == 1
    assert "tool_choice" not in seen[0]


def test_other_400_still_raises(monkeypatch):
    def fake_post(self, url, json=None, headers=None, timeout=None):
        return FakeResp(400, '{"error":{"message":"bad model"}}')

    monkeypatch.setattr(vlm.requests.Session, "post", fake_post)
    with pytest.raises(vlm.VLMError, match="HTTP 400"):
        vlm.VLMClient("https://x", "m", "k").chat([])
