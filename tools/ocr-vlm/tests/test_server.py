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


def _display_png(path, w=900, h=220, x0=100, x1=700, y0=40, y1=180,
                 total=12, glyphs=""):
    """A synthetic frame the segmenter can localize (bright bg, dark band)."""
    img = np.full((h, w, 3), 170, np.uint8)
    img[y0:y1, x0:x1] = 25
    pitch = (x1 - x0) / total
    for i in range(1, total):
        xi = int(round(x0 + i * pitch))
        img[y0:y1, xi - 2:xi + 3] = 8
    cy = (y0 + y1) // 2
    for i, ch in enumerate(glyphs[:total]):
        if ch in (" ", ""):
            continue
        a = int(round(x0 + i * pitch)) + 8
        b = int(round(x0 + (i + 1) * pitch)) - 8
        img[cy - 25:cy + 25, a:b] = 255
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
    last_no_detect = False
    last_detected = None
    last_display = None
    last_blanks = None
    last_composed = None

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

    assert c["image_mode"] == "strip"              # default segmentation
    c = client.post("/api/config", json={"image_mode": "CELLS"}).json()
    assert c["image_mode"] == "cells"              # normalized lower-case
    assert client.post("/api/config",
                       json={"image_mode": "grid"}).status_code == 400
    for mode in ("montage", "cells-detect", "strip-detect"):
        c = client.post("/api/config", json={"image_mode": mode}).json()
        assert c["image_mode"] == mode


def test_preprocess_and_debug_config(client):
    c = client.get("/api/config").json()
    assert c["preprocess"] == "none"               # default
    assert c["debug_images"] is False

    for style in ("contrast", "invert", "binary"):
        c = client.post("/api/config", json={"preprocess": style}).json()
        assert c["preprocess"] == style
    assert client.post("/api/config",
                       json={"preprocess": "sepia"}).status_code == 400

    c = client.post("/api/config", json={"debug_images": True}).json()
    assert c["debug_images"] is True
    c = client.post("/api/config", json={"debug_images": False}).json()
    assert c["debug_images"] is False


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


def test_dataset_preview_endpoint(client, tmp_path):
    ds = tmp_path / "ds-preview"
    ds.mkdir()
    _display_png(ds / "sw_1_f1.png", glyphs="AB")
    _png(ds / "plain.png")                     # nothing to detect
    (ds / "reads.jsonl").write_text("", encoding="utf-8")
    _configure(client, ds)

    r = client.get("/api/dataset/preview/sw_1_f1.png")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    assert r.headers["X-Seg-Detected"] == "1"
    assert r.headers["X-Seg-Blanks"] == "10"

    r = client.get("/api/dataset/preview/plain.png",
                   params={"style": "binary"})
    assert r.status_code == 200
    assert r.headers["X-Seg-Detected"] == "0"  # raw photo returned instead

    assert client.get("/api/dataset/preview/nope.png").status_code == 404
    assert client.get(
        "/api/dataset/preview/..%5Csw_1_f1.png").status_code == 400


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
    assert state["pipeline"]["parts"]["read_mode"] == "auto"

    report = client.get("/api/run/report").json()
    assert report["status"] == "done"
    assert report["counts"]["total"] == 3
    assert report["counts"]["text_fallback"] == 0
    assert report["counts"]["no_detect"] == 0
    assert isinstance(report["counts"]["avg_time_s"], (int, float))
    assert report["config"]["read_mode"] == "auto"
    assert report["config"]["ocr_prompt"] == "OCR:"
    assert report["config"]["image_mode"] == "strip"
    assert report["config"]["preprocess"] == "none"
    assert report["config"]["debug_images"] is False
    assert len(report["dataset"]["sha256"]) == 64
    assert report["config"]["llm_model"] == "test-model"
    # The report records the whole reading pipeline, not just the model:
    # enough to recreate it in code.
    assert report["pipeline"]["parts"]["read_mode"] == "auto"
    assert report["pipeline"]["parts"]["segmentation"] == "strip"
    assert report["pipeline"]["parts"]["preprocess"] == "none"
    assert report["pipeline"]["parts"]["module_count"] == 12
    assert report["pipeline"]["parts"]["text_path"]["image_max_width"] == 1024
    assert report["pipeline"]["parts"]["text_path"]["image_quality"] == 80
    assert report["pipeline"]["parts"]["text_path"]["image_format"] == "jpeg"
    assert report["pipeline"]["parts"]["text_path"]["ocr_prompt"] == "OCR:"
    assert report["pipeline"]["parts"]["tool_path"] == {
        "annotated": True, "image_max_width": 1024, "image_quality": 80,
        "image_format": "jpeg"}
    assert report["pipeline"]["text"].startswith("read=auto · seg=strip")
    assert 'prompt="OCR:"' in report["pipeline"]["text"]

    runs = client.get("/api/runs").json()["runs"]
    assert runs and runs[0]["run"] == "run-001"
    assert runs[0]["vlm_total"] == 2 and runs[0]["saw_total"] == 1
    assert runs[0]["pipeline"].startswith("read=auto")
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


def test_run_with_detected_reader_metadata_and_debug_images(
        client, dataset_dir, monkeypatch):
    _configure(client, dataset_dir, image_mode="montage", preprocess="binary",
               debug_images=True)
    reader = FakeReader(W12)
    reader.last_mode = "text"
    reader.last_detected = True
    reader.last_display = {"x0": 160, "width": 600}
    reader.last_blanks = [True] * 12
    reader.last_composed = np.full((8, 8, 3), 7, np.uint8)
    monkeypatch.setattr(server, "_make_reader", lambda cfg: reader)

    assert client.post("/api/run/start", json={"limit": 1}).status_code == 200
    assert _wait_status()["status"] == "done"
    row = client.get("/api/run/rows").json()["rows"][0]
    assert row["segmentation"]["detected"] is True
    assert row["segmentation"]["blanks"] == 12
    assert row["segmentation"]["display"]["width"] == 600
    assert row["composed"].startswith("composed")
    run_dir = os.path.join(server.data_dir(), "runs", "run-001")
    assert os.path.isfile(os.path.join(run_dir, row["composed"]))
    r = client.get("/api/run/row/0/composed")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert client.get("/api/run/row/1/composed").status_code == 404

    report = client.get("/api/run/report").json()
    assert report["config"]["preprocess"] == "binary"
    assert report["config"]["debug_images"] is True


def test_row_flags_no_detect_and_report_counts_it(
        client, dataset_dir, monkeypatch):
    _configure(client, dataset_dir, image_mode="montage")
    reader = FakeReader(W12)
    reader.last_no_detect = True
    reader.last_detected = False
    monkeypatch.setattr(server, "_make_reader", lambda cfg: reader)

    assert client.post("/api/run/start", json={"limit": 1}).status_code == 200
    state = _wait_status()
    assert state["status"] == "done"
    row = client.get("/api/run/rows").json()["rows"][0]
    assert "no-detect" in row["flags"]
    assert row["segmentation"] == {"detected": False, "display": None}
    report = client.get("/api/run/report").json()
    assert report["counts"]["no_detect"] == 1


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


# -- baseline sets (curation API + verified-only benchmark flow) ---------------

def _baseline_source(tmp_path, n_photos: int = 2):
    """A calib-vlm style run dir to import as a baseline set."""
    src = tmp_path / "run-src"
    src.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(1, n_photos + 1):
        photo = f"cur_{i}_f{i}.png"
        _png(src / photo)
        frames.append({"tag": f"cur_{i}", "frameId": i,
                       "frame": "A" * 12, "photo": photo,
                       "read": "A" * 12})
    (src / "report.json").write_text(json.dumps({"frames": frames}),
                                      encoding="utf-8")
    return src


def test_verified_only_config_roundtrip(client):
    assert client.get("/api/config").json()["verified_only"] is True
    assert client.post("/api/config",
                       json={"verified_only": False}
                       ).json()["verified_only"] is False
    assert client.post("/api/config",
                       json={"verified_only": True}
                       ).json()["verified_only"] is True


def test_baseline_routes(client, tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_VLM_BASELINES", str(tmp_path / "baselines"))
    src = _baseline_source(tmp_path)

    assert client.get("/api/baselines").json()["sets"] == []

    r = client.post("/api/baselines/create",
                    json={"source_dir": str(src), "name": "curated"})
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 2
    assert r.json()["stats"] == {"total": 2, "verified": 0, "pending": 2}

    sets = client.get("/api/baselines").json()["sets"]
    assert sets[0]["name"] == "curated"
    assert sets[0]["stats"]["pending"] == 2

    info = client.get("/api/baselines/curated").json()
    photo = info["entries"][0]["photo"]
    assert info["entries"][0]["content"] == "A" * 12   # pre-filled
    assert info["entries"][0]["status"] == "pending"

    r = client.get(f"/api/baselines/curated/photo/{photo}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert client.get(
        "/api/baselines/curated/photo/nope.png").status_code == 404
    assert client.get(
        "/api/baselines/curated/photo/..%5Cx.png").status_code == 400

    # saving content verifies the entry
    r = client.post("/api/baselines/curated/entry",
                    json={"photo": photo, "content": "B" * 12})
    assert r.status_code == 200, r.text
    assert r.json()["entry"]["status"] == "verified"
    assert r.json()["stats"] == {"total": 2, "verified": 1, "pending": 1}

    # and it can be pushed back to pending while re-checking
    r = client.post("/api/baselines/curated/entry",
                    json={"photo": photo, "status": "pending"})
    assert r.json()["entry"]["status"] == "pending"
    client.post("/api/baselines/curated/entry",
                json={"photo": photo, "content": "B" * 12})

    # guards
    assert client.post("/api/baselines/curated/entry",
                       json={}).status_code == 400
    assert client.post("/api/baselines/curated/entry",
                       json={"photo": "nope.png"}).status_code == 400
    assert client.post("/api/baselines/create",
                       json={"source_dir": str(src),
                             "name": "curated"}).status_code == 400
    assert client.get("/api/baselines/noset").status_code == 400

    # per-image remove keeps a live set, whole-set discard removes it
    r = client.post("/api/baselines/curated/remove-image",
                    json={"photo": photo})
    assert r.status_code == 200
    assert r.json()["stats"]["total"] == 1
    assert client.delete("/api/baselines/curated").status_code == 200
    assert client.get("/api/baselines").json()["sets"] == []
    assert client.delete("/api/baselines/curated").status_code == 400


def test_baseline_benchmark_flow(client, tmp_path, monkeypatch):
    """A run against a baseline set scores curated content as truth,
    keeps the prior read as the baseline column, and skips pending rows."""
    monkeypatch.setenv("OCR_VLM_BASELINES", str(tmp_path / "baselines"))
    src = _baseline_source(tmp_path, n_photos=2)
    client.post("/api/baselines/create",
                json={"source_dir": str(src), "name": "curated"})
    info = client.get("/api/baselines/curated").json()
    photo = info["entries"][0]["photo"]
    client.post("/api/baselines/curated/entry",
                json={"photo": photo, "content": "B" * 12})

    # Point the benchmark at the set folder; reads.jsonl resolves to
    # the set's baseline.jsonl automatically.
    _configure(client, tmp_path / "baselines" / "curated")
    d = client.post("/api/dataset/validate").json()
    assert d["kind"] == "baseline"
    assert d["runnable"] == 1 and d["skipped_pending"] == 1
    assert d["verified"] == 1 and d["pending"] == 1

    monkeypatch.setattr(server, "_make_reader",
                        lambda cfg: FakeReader("B" * 12))
    r = client.post("/api/run/start", json={})
    assert r.status_code == 200, r.text
    assert _wait_status()["status"] == "done"

    rows = client.get("/api/run/rows").json()["rows"]
    assert len(rows) == 1                        # pending row excluded
    assert rows[0]["want"] == "B" * 12           # curated content is truth
    assert rows[0]["saw"] == "A" * 12            # prior read is the baseline
    assert rows[0]["mm_vlm"] == 0 and rows[0]["mm_saw"] == 12

    report = client.get("/api/run/report").json()
    assert report["dataset"]["kind"] == "baseline"
    assert report["dataset"]["skipped_pending"] == 1
    assert report["dataset"]["verified_only"] is True


def test_run_on_all_pending_set_fails_clearly(client, tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_VLM_BASELINES", str(tmp_path / "baselines"))
    src = _baseline_source(tmp_path, n_photos=1)
    client.post("/api/baselines/create",
                json={"source_dir": str(src), "name": "allpending"})
    _configure(client, tmp_path / "baselines" / "allpending")
    monkeypatch.setattr(server, "_make_reader",
                        lambda cfg: FakeReader("A" * 12))

    r = client.post("/api/run/start", json={})
    assert r.status_code == 200
    assert _wait_status()["status"] == "failed"
    report = client.get("/api/run/report").json()
    assert "no verified entries" in report["reason"]


def _classify_set(tmp_path, monkeypatch, client):
    """Curated set of rectangle-glyph photos + a trained local bank."""
    monkeypatch.setenv("OCR_VLM_BASELINES", str(tmp_path / "baselines"))
    src = tmp_path / "cls-src"
    src.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in (1, 2):
        photo = f"cls_{i}_f{i}.png"
        _display_png(src / photo, glyphs="A" * 12)
        frames.append({"tag": f"cls_{i}", "frameId": i, "frame": "A" * 12,
                       "photo": photo, "read": "A" * 12})
    (src / "report.json").write_text(json.dumps({"frames": frames}),
                                     encoding="utf-8")
    r = client.post("/api/baselines/create",
                    json={"source_dir": str(src), "name": "clsset"})
    assert r.status_code == 200, r.text
    info = client.get("/api/baselines/clsset").json()
    for entry in info["entries"]:
        r = client.post("/api/baselines/clsset/entry",
                        json={"photo": entry["photo"], "content": "A" * 12})
        assert r.status_code == 200, r.text
    from ocr_vlm import classifier, glyphs
    assert glyphs.build()["cells"] == 24
    classifier.train_bank()
    return tmp_path / "baselines" / "clsset"


def test_read_mode_classify_runs_without_a_provider(client, tmp_path,
                                                    monkeypatch):
    """read_mode=classify end to end: local model, no llm_* configured."""
    setdir = _classify_set(tmp_path, monkeypatch, client)
    r = client.post("/api/config", json={
        "dataset_dir": str(setdir), "dataset_file": "baseline.jsonl",
        "read_mode": "classify", "module_count": 12, "concurrency": 1,
        "charset": server.DEFAULT_CHARSET})
    assert r.status_code == 200, r.text
    assert r.json()["read_mode"] == "classify"

    d = client.post("/api/dataset/validate").json()
    assert d["runnable"] == 2 and d["kind"] == "baseline"

    r = client.post("/api/run/start", json={})
    assert r.status_code == 200, r.text
    state = _wait_status()
    assert state["status"] == "done", state
    assert state["errors"] == 0

    rows = client.get("/api/run/rows").json()["rows"]
    assert len(rows) == 2
    for row in rows:
        assert row["read"] == "A" * 12
        assert row["mm_vlm"] == 0
        assert row["read_mode"] == "classify"
        assert row["error"] is None

    state = server.harness.state()
    assert state["pipeline"]["text"].startswith("read=classify")
    parts = state["pipeline"]["parts"]
    assert parts["classifier"]["backend"] == "bank"
    assert len(parts["classifier"]["sha256"]) == 64
    assert parts["segmentation"] == "display+canonical64"

    report = client.get("/api/run/report").json()
    assert report["config"]["read_mode"] == "classify"
    assert report["config"]["classifier_backend"] == "auto"
    assert "low_conf_rows" in report["counts"]
    assert report["pipeline"]["parts"]["classifier"]["backend"] == "bank"


def test_classify_mode_without_artifact_fails_with_hint(client, tmp_path):
    ds = _make_dataset(tmp_path / "ds-nomodel", n_rows=1)
    _configure(client, ds, read_mode="classify")
    r = client.post("/api/run/start", json={})
    assert r.status_code == 400
    assert "build one first" in r.json()["detail"]


def test_classifier_config_validation(client):
    assert client.post("/api/config",
                       json={"classifier_backend": "bogus"}
                       ).status_code == 400
    assert client.post("/api/config",
                       json={"classifier_min_conf": 2}).status_code == 400
    r = client.post("/api/config", json={"classifier_backend": "cnn",
                                         "classifier_min_conf": 0.7,
                                         "classifier_min_margin": 0.2})
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["classifier_backend"] == "cnn"
    assert c["classifier_min_conf"] == 0.7
    assert c["classifier_min_margin"] == 0.2
