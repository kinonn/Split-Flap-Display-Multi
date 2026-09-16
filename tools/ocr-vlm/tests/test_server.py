"""Server tests: config, dataset endpoints, and a fake-reader run.

The VLM is replaced through the ``server._make_reader`` seam, so the
whole harness (thread, scoring, report writing, artifacts) runs without
network access.
"""

from __future__ import annotations

import json
import os
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ocr_vlm import server

W12 = "A" * 12


def _png(path, w=64, h=24):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


class FakeReading:
    def __init__(self, text: str):
        self.text = text
        self.realigned = False
        self.raw_count = len(text)
        self.warnings: list[str] = []
        self.modules = [
            {"module": i, "char": ch, "condition": "clean", "confidence": 0.9,
             "source": "vlm", "expected": " "}
            for i, ch in enumerate(text)
        ]


class FakeReader:
    """Returns a fixed text; optionally sleeps (for abort tests).

    Mimics the mode metadata the real reader exposes after every read.
    """

    last_mode = "tool"
    last_raw = None
    last_fallback = False
    last_empty = False

    def __init__(self, text: str, delay_s: float = 0.0):
        self.text = text
        self.delay_s = delay_s
        self.vlm = None

    def read(self, jpeg, total, expected="", charset="", drum=""):
        if self.delay_s:
            time.sleep(self.delay_s)
        text = self.text[:total].ljust(total, " ")
        return FakeReading(text)


class TextModeFakeReader(FakeReader):
    """A reader that fell back to (or was set to) the OCR text path."""

    last_mode = "text"
    last_raw = "A A A"
    last_fallback = True


def _make_dataset(directory, n_rows=3):
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(1, n_rows + 1):
        photo = f"sw_{i}_f{i}.png"
        if i != n_rows:  # last photo is deliberately missing
            _png(directory / photo)
        saw = W12 if i != 2 else "A" * 11 + "X"
        rows.append({"photo": photo, "want": W12, "saw": saw})
    (directory / "reads.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return directory


@pytest.fixture
def dataset_dir(tmp_path):
    return _make_dataset(tmp_path / "ds", n_rows=3)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_VLM_DATA", str(tmp_path / "data"))
    for var in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY",
                "OCR_VLM_DATASET_DIR", "OCR_VLM_DATASET_FILE"):
        monkeypatch.delenv(var, raising=False)
    with server.harness.lock:
        server.harness.status = "idle"
        server.harness.rows = []
        server.harness.events = []
        server.harness.report = None
        server.harness.dataset_desc = {}
        server.harness.abort_event = threading.Event()
    return TestClient(server.app)


def _configure(client, dataset_dir, api_key="sk-test1234", **extra):
    payload = {
        "dataset_dir": str(dataset_dir),
        "dataset_file": "reads.jsonl",
        "llm_base_url": "https://api.openai.com/v1",
        "llm_model": "test-model",
        "llm_api_key": api_key,
        "module_count": 12,
        "concurrency": 1,
        "charset": server.DEFAULT_CHARSET,
    }
    payload.update(extra)
    r = client.post("/api/config", json=payload)
    assert r.status_code == 200, r.text


def _wait_status(timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = server.harness.state()
        if st["status"] in ("done", "aborted", "failed"):
            return st
        time.sleep(0.05)
    raise AssertionError(f"run did not finish: {server.harness.state()}")


def _wait_done_at_least(n, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.harness.state()["done"] >= n:
            return
        time.sleep(0.02)
    raise AssertionError("rows did not complete in time")


def test_config_roundtrip_and_masking(client):
    _configure(client, "/tmp/whatever", api_key="sk-abcd9876",
               concurrency=99, module_count=1000)
    c = client.get("/api/config").json()
    assert c["llm_api_key"] == "***9876"          # masked, never raw
    assert c["dataset_dir"] == "/tmp/whatever"
    assert c["concurrency"] == 8                   # clamped to max
    assert c["module_count"] == 64                 # clamped to max
    assert len(c["charset"]) == 48


def test_read_mode_config(client):
    c = client.get("/api/config").json()
    assert c["read_mode"] == "auto"                # default
    assert c["ocr_prompt"] == "OCR:"

    r = client.post("/api/config", json={"read_mode": "TEXT",
                                          "ocr_prompt": ""})
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["read_mode"] == "text"                # normalized
    assert c["ocr_prompt"] == "OCR:"               # empty -> default

    r = client.post("/api/config", json={"read_mode": "bogus"})
    assert r.status_code == 400
    assert "read_mode" in r.json()["detail"]


def test_image_config_defaults_and_validation(client):
    c = client.get("/api/config").json()
    assert c["image_max_width"] == 1024
    assert c["image_quality"] == 80
    assert c["image_format"] == "jpeg"
    assert c["ocr_max_tokens"] == 128              # default cap

    r = client.post("/api/config", json={"image_max_width": 1280,
                                          "image_quality": 95,
                                          "image_format": "PNG"})
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["image_max_width"] == 1280
    assert c["image_quality"] == 95
    assert c["image_format"] == "png"              # normalized lower-case

    assert client.post("/api/config",
                       json={"image_format": "bmp"}).status_code == 400
    c = client.post("/api/config", json={"image_max_width": 99999,
                                          "image_quality": 999}).json()
    assert c["image_max_width"] == 4096            # clamped
    assert c["image_quality"] == 100               # clamped

    c = client.post("/api/config", json={"ocr_max_tokens": 0}).json()
    assert c["ocr_max_tokens"] == 0                # 0 = uncapped, kept
    c = client.post("/api/config", json={"ocr_max_tokens": 99999}).json()
    assert c["ocr_max_tokens"] == 4096             # clamped


def test_dataset_validate_and_photo_endpoints(client, dataset_dir):
    _configure(client, dataset_dir)
    d = client.post("/api/dataset/validate").json()
    assert d["rows"] == 3 and d["runnable"] == 3
    assert d["missing_count"] == 1
    assert d["missing_photos"] == ["sw_3_f3.png"]
    assert d["flagged_count"] == 1
    assert d["prefixes"] == [{"prefix": "sw", "rows": 3}]

    r = client.get("/api/dataset/photo/sw_1_f1.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert client.get("/api/dataset/photo/nope.png").status_code == 404
    assert client.get(
        "/api/dataset/photo/..%5Csw_1_f1.png").status_code == 400


def test_validate_requires_directory(client):
    r = client.post("/api/dataset/validate")
    assert r.status_code == 400
    assert "dataset directory" in r.json()["detail"]


def test_run_with_fake_reader(client, dataset_dir, monkeypatch):
    _configure(client, dataset_dir)
    # Every read returns 11 A + B: row 1 scores 1 mm vs want (baseline 0),
    # row 2 scores 1 mm (baseline 1 too), row 3 is the missing photo.
    monkeypatch.setattr(server, "_make_reader",
                        lambda cfg: FakeReader("A" * 11 + "B"))

    r = client.post("/api/run/start", json={})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "running"
    state = _wait_status()
    assert state["status"] == "done"
    assert state["total"] == 3 and state["done"] == 3
    assert state["errors"] == 1

    rows = client.get("/api/run/rows").json()["rows"]
    assert len(rows) == 3
    assert "modules" not in rows[0]                  # compact for the table
    assert rows[0]["mm_vlm"] == 1 and rows[0]["mm_saw"] == 0
    assert rows[0]["read_mode"] == "tool"            # fake defaults
    assert rows[0]["raw"] is None and rows[0]["fallback"] is False
    assert rows[1]["mm_vlm"] == 1 and rows[1]["mm_saw"] == 1
    assert rows[2]["error"] == "photo not found"

    detail = client.get("/api/run/row/0").json()
    assert len(detail["modules"]) == 12
    assert client.get("/api/run/row/99").status_code == 404

    summary = state["summary"]
    assert summary["vlm"]["total"] == 2
    assert summary["saw"]["total"] == 1
    assert summary["head_to_head"] == {"n": 2, "better": 0, "equal": 1,
                                       "worse": 1, "delta_mean": -0.5}

    report = client.get("/api/run/report").json()
    assert report["status"] == "done"
    assert report["counts"]["total"] == 3
    assert report["counts"]["text_fallback"] == 0
    assert isinstance(report["counts"]["avg_time_s"], (int, float))
    assert report["config"]["read_mode"] == "auto"
    assert report["config"]["ocr_prompt"] == "OCR:"
    assert len(report["dataset"]["sha256"]) == 64
    assert report["config"]["llm_model"] == "test-model"

    runs = client.get("/api/runs").json()["runs"]
    assert runs and runs[0]["run"] == "run-001"
    assert runs[0]["vlm_total"] == 2 and runs[0]["saw_total"] == 1
    assert isinstance(runs[0]["avg_time_s"], (int, float))

    run_dir = os.path.join(server.data_dir(), "runs", "run-001")
    assert os.path.isfile(os.path.join(run_dir, "results.json"))
    assert os.path.isfile(os.path.join(run_dir, "report.json"))
    assert os.path.isfile(os.path.join(run_dir, "events.jsonl"))


def test_run_rows_carry_text_mode_metadata(client, dataset_dir, monkeypatch):
    _configure(client, dataset_dir)
    monkeypatch.setattr(server, "_make_reader",
                        lambda cfg: TextModeFakeReader(W12))
    assert client.post("/api/run/start", json={"limit": 1}).status_code == 200
    state = _wait_status()
    assert state["status"] == "done"

    row = client.get("/api/run/rows").json()["rows"][0]
    assert row["read_mode"] == "text"
    assert row["raw"] == "A A A"
    assert row["fallback"] is True
    assert "text-fallback" in row["flags"]

    report = client.get("/api/run/report").json()
    assert report["counts"]["text_fallback"] == 1
    assert report["config"]["read_mode"] == "auto"


def test_run_start_requires_credentials(client, dataset_dir):
    # No dataset configured yet.
    r = client.post("/api/run/start", json={})
    assert r.status_code == 400
    assert "dataset directory" in r.json()["detail"]

    # Dataset configured, but a remote provider without an API key.
    _configure(client, dataset_dir, api_key="")
    r = client.post("/api/run/start", json={})
    assert r.status_code == 400
    assert "llm_api_key" in r.json()["detail"]


def test_run_limit_and_abort(client, tmp_path, monkeypatch):
    big = _make_dataset(tmp_path / "big", n_rows=8)
    # Give every row a photo so the run has 8 real VLM calls queued.
    for i in range(1, 9):
        _png(big / f"sw_{i}_f{i}.png")
    _configure(client, big)
    monkeypatch.setattr(server, "_make_reader",
                        lambda cfg: FakeReader(W12, delay_s=0.25))

    r = client.post("/api/run/start", json={"limit": "3"})
    assert r.status_code == 200, r.text
    state = _wait_status()
    assert state["status"] == "done"
    assert state["total"] == 3                       # limit applied

    r = client.post("/api/run/start", json={})
    assert r.status_code == 200
    _wait_done_at_least(1)
    r = client.post("/api/run/abort")
    assert r.status_code == 200
    state = _wait_status()
    assert state["status"] == "aborted"
    assert state["done"] < state["total"]

    # A fresh start while idle is allowed again.
    r = client.post("/api/run/start", json={"limit": 1})
    assert r.status_code == 200
    assert _wait_status()["status"] == "done"
