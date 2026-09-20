"""Minimal OpenAI-compatible vision chat client (no SDK dependency).

One code path covers OpenAI, OpenRouter, Ollama, vLLM and any other
OpenAI-compatible gateway: POST {base_url}/chat/completions with text +
base64-JPEG image parts and native function-calling `tools`.
"""

from __future__ import annotations

import base64
import time
import uuid

import requests

USER_AGENT = "splitflap-calib-auto/0.1.0"

# Transient provider/network failures are retried with exponential backoff:
# a single 429, 5xx or dropped connection must not poison a calibration
# frame (the calibrator degrades a failed read to an all-`unreadable`
# reading, which escalates and can trip the unreliable-reads abort).
# Anything else that is not 200 still fails immediately.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
RETRY_ATTEMPTS = 4
RETRY_BACKOFF_S = 2.0
RETRY_BACKOFF_MAX_S = 30.0


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
                 session_id: str | None = None,
                 max_attempts: int = RETRY_ATTEMPTS,
                 backoff_s: float = RETRY_BACKOFF_S):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.extra_headers = extra_headers or {}
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_s = max(0.0, float(backoff_s))
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
        # first such rejection drops it for the rest of the client's
        # life, so the wasted round trip is paid once, not per frame.
        self._allow_forced_tool_choice = True
        # Token usage of the last chat() call ({} when unreported).
        self.last_usage: dict = {}

    def _post_chat(self, payload: dict, headers: dict):
        """POST /chat/completions, retrying transient failures.

        Retries 429/5xx and network errors with exponential backoff
        (honouring a `Retry-After` header when the provider sends one).
        A persistent failure still raises VLMError after `max_attempts`,
        so a provider that is genuinely down stops the run instead of
        stalling it.
        """
        delay = self.backoff_s
        last = "LLM request failed"
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self.session.post(
                    self.base_url + "/chat/completions", json=payload,
                    headers=headers, timeout=self.timeout_s)
            except requests.RequestException as exc:
                last = f"LLM request failed: {exc}"
                if attempt >= self.max_attempts:
                    raise VLMError(last) from exc
                time.sleep(delay)
                delay = min(delay * 2.0, RETRY_BACKOFF_MAX_S)
                continue
            if (resp.status_code in RETRY_STATUS
                    and attempt < self.max_attempts):
                wait = delay
                header = getattr(resp, "headers", None) or {}
                try:
                    retry_after = float(header.get("Retry-After", ""))
                    wait = max(wait, min(retry_after, RETRY_BACKOFF_MAX_S))
                except (TypeError, ValueError):
                    pass
                time.sleep(wait)
                delay = min(delay * 2.0, RETRY_BACKOFF_MAX_S)
                continue
            return resp
        raise VLMError(last)

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str | None = None,
             max_tokens: int | None = None) -> dict:
        """Returns the assistant message: {"content": str|None,
        "tool_calls": [{"id": str, "name": str, "arguments": dict}]}.

        ``max_tokens`` caps this one generation (None = server default).
        Callers that only need a short answer (e.g. a 12-character OCR
        read) should cap it: a model that degenerates into a repetition
        loop — observed with greedy decoding on rows of one repeated
        character — otherwise runs to the server's default limit, which
        costs seconds per frame for output nobody reads.
        """
        import json

        payload: dict = {"model": self.model, "messages": messages}
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        if tools:
            payload["tools"] = tools
            if self._allow_forced_tool_choice:
                payload["tool_choice"] = tool_choice or "auto"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "User-Agent": USER_AGENT}
        if "opencode.ai" in self.base_url:
            headers["X-Opencode-Session"] = f"splitflap-calib-auto-{self.session_id}"
        headers.update(self.extra_headers)
        resp = self._post_chat(payload, headers)
        if resp.status_code == 400 and "tool_choice" in resp.text.lower():
            # Thinking-mode provider rejected the forced choice: retry
            # once without it and remember, so later frames go straight
            # through (the tool schema still guides the model).
            self._allow_forced_tool_choice = False
            payload.pop("tool_choice", None)
            resp = self._post_chat(payload, headers)
        if resp.status_code != 200:
            raise VLMError(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            body = resp.json()
            msg = body["choices"][0]["message"]
        except (KeyError, IndexError, ValueError) as exc:
            raise VLMError(f"bad LLM response: {resp.text[:300]}") from exc
        # Token usage for cost/speed analysis (may be absent per provider).
        try:
            usage = body.get("usage") or {}
            self.last_usage = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                "completion_tokens": int(usage.get("completion_tokens", 0)),
            }
        except (TypeError, ValueError):
            self.last_usage = {}
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
