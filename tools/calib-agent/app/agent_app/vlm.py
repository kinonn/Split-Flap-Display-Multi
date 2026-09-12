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
        # One connection pool for the whole run: bare requests.post()
        # performs a fresh TCP+TLS handshake per call, which adds up over
        # dozens of turns.
        self.session = requests.Session()
        # Some providers (thinking models) reject a forced tool_choice
        # with "Thinking mode does not support this tool_choice": the
        # first such rejection drops it for the rest of the client's life.
        self._allow_forced_tool_choice = True

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Returns the assistant message: {"content": str|None,
        "tool_calls": [{"id": str, "name": str, "arguments": dict}]}."""
        import json

        payload: dict = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            if self._allow_forced_tool_choice:
                payload["tool_choice"] = "auto"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "User-Agent": USER_AGENT}
        if "opencode.ai" in self.base_url:
            headers["X-Opencode-Session"] = f"splitflap-calib-{self.session_id}"
        headers.update(self.extra_headers)
        try:
            resp = self.session.post(self.base_url + "/chat/completions", json=payload,
                                     headers=headers, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise VLMError(f"LLM request failed: {exc}") from exc
        if resp.status_code == 400 and "tool_choice" in resp.text.lower():
            # Thinking-mode provider rejected the forced choice: retry
            # once without it and remember, so later turns go straight
            # through (the tool schema still guides the model).
            self._allow_forced_tool_choice = False
            payload.pop("tool_choice", None)
            try:
                resp = self.session.post(self.base_url + "/chat/completions",
                                         json=payload, headers=headers,
                                         timeout=self.timeout_s)
            except requests.RequestException as exc:
                raise VLMError(f"LLM request failed: {exc}") from exc
        if resp.status_code != 200:
            raise VLMError(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            msg = resp.json()["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as exc:
            raise VLMError(f"bad LLM response: {resp.text[:300]}") from exc
        calls = []
        for i, call in enumerate(msg.get("tool_calls") or []):
            fn = call.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {"_parse_error": fn.get("arguments", "")}
            # A missing/null id would never match its tool response and
            # strict providers reject the history for it — synthesize one.
            calls.append({"id": call.get("id") or f"call_{i}",
                          "name": fn.get("name", ""),
                          "arguments": args if isinstance(args, dict) else {}})
        return {"content": msg.get("content"), "tool_calls": calls}
