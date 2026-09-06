"""HTTP client for the split-flap calibration APIs (no browser needed).

Master-only access: fleet-wide shows and remote offset persists all go
through the master at one hostname/IP. Preview is local-controller only
in firmware, so remote groups are tuned via persist-verify-revert.
"""

from __future__ import annotations

import time

import requests


class CalibError(RuntimeError):
    pass


class Display:
    def __init__(self, host: str, timeout_s: float = 10.0, settle_timeout_s: float = 300.0):
        host = host.strip()
        if "://" not in host:
            host = "http://" + host
        self.base = host.rstrip("/")
        self.session = requests.Session()
        # Per-request timeout (fast fail on unreachable host); the overall
        # move-settle deadline lives in wait_settled/show_and_settle.
        self.timeout_s = timeout_s
        self.settle_timeout_s = settle_timeout_s

    def _get(self, path: str, **kwargs):
        try:
            resp = self.session.get(self.base + path, timeout=self.timeout_s, **kwargs)
        except requests.RequestException as exc:
            raise CalibError(f"GET {path} failed: {exc}") from exc
        if resp.status_code != 200:
            raise CalibError(f"GET {path} -> HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def _post(self, path: str, payload: dict, ok=(200, 202)):
        try:
            resp = self.session.post(self.base + path, json=payload, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise CalibError(f"POST {path} failed: {exc}") from exc
        if resp.status_code not in ok:
            raise CalibError(f"POST {path} -> HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    # -- calibration endpoints ------------------------------------------------
    def status(self) -> dict:
        return self._get("/api/calib/status")

    def contract(self) -> dict:
        return self._get("/calib-contract.json")

    def hold(self, active: bool) -> dict:
        return self._post("/api/calib/hold", {"active": bool(active)})

    def show(self, frame: str, dwell_ms: int = 800) -> dict:
        return self._post("/api/calib/show", {"frame": frame, "dwellMs": dwell_ms})

    def frame_info(self, frame_id: int) -> dict:
        return self._get("/api/calib/frame", params={"frameId": frame_id})

    def preview(self, module: int, char_index: int, delta: int) -> dict:
        return self._post(
            "/api/calib/preview",
            {"module": module, "charIndex": char_index, "delta": delta},
        )

    def persist(self, scope, kind: str, value: int, module: int = 0, char_index: int = 0) -> dict:
        payload: dict = {"scope": scope, "kind": kind, "value": value}
        if kind in ("char", "module"):
            payload["module"] = module
        if kind == "char":
            payload["charIndex"] = char_index
        return self._post("/api/calib/offsets", payload)

    # -- snapshot / rollback (plain settings API) ------------------------------
    def snapshot(self) -> dict:
        """Full GET /settings body; POST its inner `settings` to roll back."""
        try:
            resp = self.session.get(self.base + "/settings", timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise CalibError(f"GET /settings failed: {exc}") from exc
        if resp.status_code != 200:
            raise CalibError(f"GET /settings -> HTTP {resp.status_code}")
        return resp.json()

    def restore(self, snapshot: dict) -> dict:
        settings = snapshot.get("settings", snapshot)
        try:
            resp = self.session.post(self.base + "/settings", json=settings, timeout=self.timeout_s)
        except requests.RequestException as exc:
            raise CalibError(f"POST /settings failed: {exc}") from exc
        if resp.status_code != 200:
            raise CalibError(f"POST /settings -> HTTP {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    # -- helpers ----------------------------------------------------------------
    def wait_settled(self, timeout_s: float | None = None, poll_s: float = 0.5) -> dict:
        """Poll status until busy==false; returns the final status."""
        deadline = time.monotonic() + (self.settle_timeout_s if timeout_s is None else timeout_s)
        last: dict = {}
        while True:
            last = self.status()
            if not last.get("busy", False):
                return last
            if time.monotonic() > deadline:
                raise CalibError("display stayed busy past timeout")
            time.sleep(poll_s)

    def show_and_settle(self, frame: str, dwell_ms: int = 800, timeout_s: float | None = None) -> dict:
        """Show a frame and wait until the display reports settled."""
        show_resp = self.show(frame, dwell_ms)
        frame_id = show_resp["frameId"]
        self.wait_settled(timeout_s=timeout_s)
        info = self.frame_info(frame_id)
        if not info.get("settled", False):
            raise CalibError(f"frame {frame_id} never reported settled")
        return {"frameId": frame_id, "fleetFrame": show_resp.get("fleetFrame", False)}
