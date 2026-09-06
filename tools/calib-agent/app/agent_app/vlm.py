"""Minimal OpenAI-compatible vision chat client (no SDK dependency).

One code path covers OpenAI, OpenRouter, Ollama, vLLM and any other
OpenAI-compatible gateway: POST {base_url}/chat/completions with text +
base64-JPEG image parts and native function-calling `tools`.
"""

from __future__ import annotations

import base64
import uuid

import requests

USER_AGENT = "splitflap-calib-agent/0.1.0"


class VLMError(RuntimeError):
    pass


def image_part(jpeg_bytes: bytes) -> dict:
    return {"type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()}}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


class VLMClient:
    def __init__(self, base_url: str, model: str, api_key: str,
                 timeout_s: float = 180.0, extra_headers: dict | None = None,
                 session_id: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.extra_headers = extra_headers or {}
        # OpenCode Go asks clients to identify themselves and send an
        # x-opencode-session header (prompt-cache optimization, abuse
        # monitoring). Only sent to opencode.ai endpoints.
        self.session_id = session_id or uuid.uuid4().hex

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Returns the assistant message: {"content": str|None,
        "tool_calls": [{"id": str, "name": str, "arguments": dict}]}."""
        import json

        payload: dict = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "User-Agent": USER_AGENT}
        if "opencode.ai" in self.base_url:
            headers["X-Opencode-Session"] = f"splitflap-calib-{self.session_id}"
        headers.update(self.extra_headers)
        try:
            resp = requests.post(self.base_url + "/chat/completions", json=payload,
                                 headers=headers, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise VLMError(f"LLM request failed: {exc}") from exc
        if resp.status_code != 200:
            raise VLMError(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            msg = resp.json()["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as exc:
            raise VLMError(f"bad LLM response: {resp.text[:300]}") from exc
        calls = []
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {"_parse_error": fn.get("arguments", "")}
            calls.append({"id": call.get("id", ""), "name": fn.get("name", ""),
                          "arguments": args if isinstance(args, dict) else {}})
        return {"content": msg.get("content"), "tool_calls": calls}
