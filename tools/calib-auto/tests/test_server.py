"""Server smoke tests: config, golden routes, benchmark flow, jobs.

The VLM is replaced through the ``server._make_bench_reader`` seam and
the trainer through ``train_cnn.train``/``save_model`` stubs, so the
whole server runs without network, GPU or a camera.
"""

from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient
from synthutil import synth_display

from calib_auto import classifier, cnn_reader, glyphs, golden, server, train_cnn
from calib_auto.reader import ModuleReading, Reading
from calib_auto.segment import DisplayBox


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_AUTO_DATA", str(tmp_path))
    for var in ("CALIB_AUTO_DISPLAY_HOST", "LLM_BASE_URL", "LLM_MODEL",
                "LLM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for harness in (server.bench, server.cal, server.cnn_test):
        with harness.lock:
            harness.status = "idle"
            harness.events = []
            harness.photos = []
            harness.report = None
            harness.run_dir = ""
            harness.abort_event = threading.Event() \
                if hasattr(harness, "abort_event") else None
        if hasattr(harness, "rows"):
            harness.rows = []
    with server.train_job.lock:
        server.train_job.status = "idle"
        server.train_job.events = []
        server.train_job.result = None
        server.train_job.run_dir = ""
    return TestClient(server.app)


def _configure(client, **extra):
    payload = {"display_host": "splitflap.local", "module_count": 12}
    payload.update(extra)
    r = client.post("/api/config", json=payload)
    assert r.status_code == 200, r.text
    return r.json()


def _make_source(tmp_path, name="src", content="A" * 12, photos=2):
    src = tmp_path / name
    src.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(photos):
        photo = f"cur_{i}_f{i}.png"
        synth_display(src / photo, content, seed=i)
        frames.append({"tag": f"cur_{i}", "frameId": i, "frame": content,
                       "photo": photo, "read": content})
    (src / "report.json").write_text(json.dumps({"frames": frames}),
                                     encoding="utf-8")
    return src


def _wait_status(harness, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = harness.state()
        if st["status"] in ("done", "aborted", "failed"):
            return st
        time.sleep(0.05)
    raise AssertionError(f"run did not finish: {harness.state()}")


# -- config --------------------------------------------------------------------

def test_config_defaults_and_masking(client):
    c = client.get("/api/config").json()
    assert c["read_mode"] == "auto"
    assert len(c["charset"]) == 48
    assert c["llm_api_key"] == ""
    _configure(client, llm_base_url="https://api.openai.com/v1",
               llm_model="m", llm_api_key="sk-abcd9876")
    c = client.get("/api/config").json()
    assert c["llm_api_key"] == "***9876"


def test_config_validation(client):
    assert client.post("/api/config",
                       json={"read_mode": "bogus"}).status_code == 400
    assert client.post("/api/config",
                       json={"classifier_backend": "bogus"}
                       ).status_code == 400
    assert client.post("/api/config",
                       json={"classifier_min_conf": 2}).status_code == 400
    assert client.post("/api/config",
                       json={"module_count": 1000}).status_code == 400
    r = client.post("/api/config", json={"read_mode": "TEXT",
                                         "module_count": 12,
                                         "classifier_min_conf": 0.7})
    assert r.status_code == 200, r.text
    assert r.json()["read_mode"] == "text"
    assert r.json()["classifier_min_conf"] == 0.7


def test_config_roundtrips_camera(client):
    c = _configure(client, brightness=61, crop_percent=12.5, warmup_s=4,
                   exposure=None)
    assert c["brightness"] == 61
    assert c["crop_percent"] == 12.5
    assert c["warmup_s"] == 4


def test_config_blank_gate_defaults_off_and_roundtrips(client):
    assert client.get("/api/config").json()["blank_gate"] is False
    c = _configure(client, blank_gate=True)
    assert c["blank_gate"] is True
    c = _configure(client, blank_gate=False)
    assert c["blank_gate"] is False


# -- golden routes ---------------------------------------------------------------

def test_golden_routes(client, tmp_path):
    assert client.get("/api/golden").json()["sets"] == []
    src = _make_source(tmp_path)
    r = client.post("/api/golden/import",
                    json={"source_dir": str(src), "name": "curated",
                          "image_format": "keep"})
    assert r.status_code == 200, r.text
    assert r.json()["count"] == 2
    sets = client.get("/api/golden").json()["sets"]
    assert sets[0]["name"] == "curated"

    info = client.get("/api/golden/curated").json()
    photo = info["entries"][0]["photo"]
    assert info["entries"][0]["content"] == "A" * 12   # pre-filled
    assert info["entries"][0]["status"] == "pending"

    r = client.get(f"/api/golden/curated/photo/{photo}")
    assert r.status_code == 200
    assert client.get(
        "/api/golden/curated/photo/nope.png").status_code == 404
    assert client.get(
        "/api/golden/curated/photo/..%5Cx.png").status_code == 400

    # saving content verifies the entry
    r = client.post("/api/golden/curated/entry",
                    json={"photo": photo, "content": "B" * 12})
    assert r.status_code == 200, r.text
    assert r.json()["entry"]["status"] == "verified"
    assert r.json()["stats"] == {"total": 2, "verified": 1, "pending": 1}

    # guards
    assert client.post("/api/golden/curated/entry",
                       json={}).status_code == 400
    assert client.post("/api/golden/curated/entry",
                       json={"photo": "nope.png"}).status_code == 400
    assert client.post("/api/golden/curated/remove-image",
                       json={"photo": photo}).status_code == 200
    assert client.delete("/api/golden/curated").status_code == 200
    assert client.get("/api/golden").json()["sets"] == []


def test_models_endpoint_empty(client):
    m = client.get("/api/models").json()
    assert m["caches"] == [] and m["bank"] is None and m["cnn"] is None
    assert m["charset"] == classifier.CHARSET  # drives the UI distribution table


def test_cal_review_lists_frames_and_boxes(client, tmp_path):
    run = tmp_path / "runs" / "cal-001"
    run.mkdir(parents=True)
    photo = "p0_allH_f1.png"
    synth_display(run / photo, "H" * 12, seed=1)
    boxes = [[i * 10, 20, (i + 1) * 10, 80] for i in range(12)]
    report = {
        "result": "converged", "reason": "ok",
        "config": {"approach": "cnn"},
        "frames": [{"tag": "p0_allH", "frameId": 1,
                     "frame": "H" * 12, "photo": photo,
                     "read": "H" * 12, "boxes": boxes,
                     "display": {"x0": 0, "y0": 20, "x1": 120,
                                 "y1": 80, "confidence": 1.0},
                     "modules": [{"module": 0, "char": "H",
                                  "confidence": 0.91,
                                  "source": "classifier"}]}],
    }
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")

    runs = client.get("/api/cal/runs")
    assert runs.status_code == 200, runs.text
    assert runs.json()["runs"] == [{
        "run": "cal-001", "result": "converged", "reason": "ok",
        "approach": "cnn", "frames": 1, "started": None,
        "finished": None, "has_boxes": True,
    }]
    frames = client.get("/api/cal/run/cal-001/frames")
    assert frames.status_code == 200
    body = frames.json()
    assert body["frames"][0]["boxes"] == boxes
    assert body["frames"][0]["modules"][0]["source"] == "classifier"
    photo_response = client.get(
        "/api/cal/run/cal-001/photo/p0_allH_f1.png")
    assert photo_response.status_code == 200
    assert client.get("/api/cal/run/other/photo/p0_allH_f1.png").status_code == 400

    # Legacy CNN reports without persisted boxes are reconstructed from the
    # original image using the same JPEG + segmentation path.
    legacy = dict(report)
    legacy["frames"] = [dict(legacy["frames"][0])]
    legacy["frames"][0].pop("boxes")
    legacy["frames"][0].pop("display")
    (run / "report.json").write_text(json.dumps(legacy), encoding="utf-8")
    legacy_frames = client.get("/api/cal/run/cal-001/frames").json()
    assert legacy_frames["frames"][0]["boxes_recreated"] is True
    assert len(legacy_frames["frames"][0]["boxes"]) == 12
    assert legacy_frames["frames"][0]["geometry_image_size"]


def test_cnn_reader_blank_gate_off_by_default(monkeypatch):
    image = __import__("numpy").zeros((100, 120, 3), dtype="uint8")
    boxes = [(0, 20, 10, 80), (10, 20, 20, 80)]
    monkeypatch.setattr(cnn_reader.segment, "find_display",
                        lambda _: DisplayBox(0, 20, 20, 80, 1.0))
    monkeypatch.setattr(cnn_reader.segment, "module_boxes",
                        lambda *_args, **_kwargs: boxes)
    monkeypatch.setattr(cnn_reader.segment, "crop_modules",
                        lambda *_args, **_kwargs: [image[:, :10], image[:, 10:20]])
    seen = {}

    def fake_blank(_crop):
        raise AssertionError("is_blank must not run when the gate is off")

    def fake_classify(_bank, _crop, *, blank, blank_gate=False, floor=170.0):
        seen["blank"] = blank
        seen["blank_gate"] = blank_gate
        return (" ", 1.0, 1.0, "classifier")

    monkeypatch.setattr(cnn_reader.segment, "is_blank", fake_blank)
    monkeypatch.setattr(cnn_reader.classifier, "classify_cell", fake_classify)
    monkeypatch.setattr(cnn_reader.segment, "grid_support", lambda *_a, **_k: 1.0)
    reader = cnn_reader.CnnReader(model=object())
    assert reader.blank_gate is False
    reading = reader.read(image, total=2)
    assert reading.text == "  "
    assert seen == {"blank": False, "blank_gate": False}
    assert reader.last_blanks == [False, False]
    assert reading.warnings == []
    assert all(m.source == "classifier" for m in reading.modules)


def test_cnn_reader_blank_gate_opt_in(monkeypatch):
    image = __import__("numpy").zeros((100, 120, 3), dtype="uint8")
    boxes = [(0, 20, 10, 80), (10, 20, 20, 80)]
    monkeypatch.setattr(cnn_reader.segment, "find_display",
                        lambda _: DisplayBox(0, 20, 20, 80, 1.0))
    monkeypatch.setattr(cnn_reader.segment, "module_boxes",
                        lambda *_args, **_kwargs: boxes)
    monkeypatch.setattr(cnn_reader.segment, "crop_modules",
                        lambda *_args, **_kwargs: [image[:, :10], image[:, 10:20]])
    monkeypatch.setattr(cnn_reader.segment, "is_blank", lambda _: True)
    monkeypatch.setattr(cnn_reader.classifier, "classify_cell",
                        lambda *_args, **_kwargs: (" ", 1.0, 1.0, "cv"))
    reader = cnn_reader.CnnReader(model=object(), blank_gate=True)
    reading = reader.read(image, total=2)
    assert reading.text == "  "
    assert reader.last_blanks == [True, True]
    assert any("blank cells decided by OpenCV" in w
               for w in reading.warnings)
    assert all(m.source == "cv" for m in reading.modules)


def test_cnn_reader_keeps_module_boxes(monkeypatch):
    image = __import__("numpy").zeros((100, 120, 3), dtype="uint8")
    boxes = [(0, 20, 10, 80), (10, 20, 20, 80)]
    monkeypatch.setattr(cnn_reader.segment, "find_display",
                        lambda _: DisplayBox(0, 20, 20, 80, 1.0))
    monkeypatch.setattr(cnn_reader.segment, "module_boxes",
                        lambda *_args, **_kwargs: boxes)
    monkeypatch.setattr(cnn_reader.segment, "crop_modules",
                        lambda *_args, **_kwargs: [image[:, :10], image[:, 10:20]])
    monkeypatch.setattr(cnn_reader.segment, "is_blank", lambda _: True)
    monkeypatch.setattr(cnn_reader.classifier, "classify_cell",
                        lambda *_args, **_kwargs: (" ", 1.0, 1.0, "cv"))
    reader = cnn_reader.CnnReader(model=object(), blank_gate=True)
    reading = reader.read(image, total=2)
    assert reading.text == "  "
    assert reader.last_boxes == boxes


# -- benchmark ------------------------------------------------------------------

class FakeBenchReader:
    """Minimal reader for the benchmark harness seam."""

    last_mode = "tool"
    last_fallback = False
    last_empty = False
    last_raw = None
    last_no_detect = False
    last_detected = None
    last_display = None
    last_blanks = None
    last_composed = None
    vlm = None

    def __init__(self, text: str):
        self.text = text

    def read(self, img, total, expected="", charset="", drum=""):
        text = self.text[:total].ljust(total, " ")
        return Reading(
            [ModuleReading(i, ch, "clean", 0.9, "vlm")
             for i, ch in enumerate(text)],
            raw_count=len(text))


def test_bench_start_guards(client):
    r = client.post("/api/bench/start", json={})
    assert r.status_code == 400
    assert "golden set" in r.json()["detail"]
    _configure(client, dataset_dir="x")
    # The default provider (opencode.ai) needs an API key; a local base
    # URL would pass this guard instead.
    r = client.post("/api/bench/start", json={})
    assert r.status_code == 400
    assert "llm_api_key" in r.json()["detail"]


def test_bench_flow_on_golden_set(client, tmp_path, monkeypatch):
    src = _make_source(tmp_path)
    client.post("/api/golden/import",
                json={"source_dir": str(src), "name": "gset",
                      "image_format": "keep"})
    info = client.get("/api/golden/gset").json()
    for entry in info["entries"]:
        client.post("/api/golden/gset/entry",
                    json={"photo": entry["photo"], "content": "A" * 12})
    set_path = golden.set_dir("gset")
    _configure(client, dataset_dir=set_path, dataset_file="labels.jsonl",
               llm_base_url="https://api.openai.com/v1", llm_model="m",
               llm_api_key="sk-test", concurrency=1)
    monkeypatch.setattr(server, "_make_bench_reader",
                        lambda cfg: FakeBenchReader("A" * 12))

    r = client.post("/api/bench/start", json={})
    assert r.status_code == 200, r.text
    state = _wait_status(server.bench)
    assert state["status"] == "done"
    assert state["report"]["summary"]["vlm"]["mean"] == 0.0
    assert state["report"]["dataset"]["kind"] == "golden"
    rows = client.get("/api/bench/rows").json()["rows"]
    assert len(rows) == 2
    assert all(r_["mm_vlm"] == 0 for r_ in rows)
    detail = client.get("/api/bench/row/0").json()
    assert len(detail["modules"]) == 12
    runs = client.get("/api/bench/runs").json()["runs"]
    assert runs and runs[0]["status"] == "done"


def test_bench_fails_clearly_on_all_pending(client, tmp_path, monkeypatch):
    src = _make_source(tmp_path, photos=1)
    client.post("/api/golden/import",
                json={"source_dir": str(src), "name": "pending",
                      "image_format": "keep"})
    _configure(client, dataset_dir=golden.set_dir("pending"),
               dataset_file="labels.jsonl",
               llm_base_url="https://api.openai.com/v1", llm_model="m",
               llm_api_key="sk-test")
    monkeypatch.setattr(server, "_make_bench_reader",
                        lambda cfg: FakeBenchReader("A" * 12))
    client.post("/api/bench/start", json={})
    state = _wait_status(server.bench)
    assert state["status"] == "failed"
    assert "no verified entries" in state["report"]["reason"]


# -- CNN test -------------------------------------------------------------------

def test_cnn_test_guards(client):
    r = client.post("/api/cnn-test/start", json={})
    assert r.status_code == 400
    assert "golden set" in r.json()["detail"]
    r = client.post("/api/cnn-test/start", json={"set": "nope"})
    assert r.status_code == 400
    assert client.post("/api/cnn-test/read", json={}).status_code == 400
    assert client.post("/api/cnn-test/read",
                       json={"set": "nope",
                             "photo": "p.png"}).status_code == 400
    # a set that exists but a photo that does not
    assert client.get("/api/cnn-test/row/0").status_code == 404
    assert client.get("/api/cnn-test/runs").json()["runs"] == []


def test_cnn_test_flow(client, tmp_path):
    content = "AMAMAMAMAMAM"
    src = tmp_path / "ct-src"
    src.mkdir()
    frames = []
    for i in (1, 2):
        photo = f"ct_{i}_f{i}.png"
        synth_display(src / photo, content, seed=i)
        frames.append({"tag": f"ct_{i}", "frameId": i, "frame": content,
                       "photo": photo, "read": content})
    (src / "report.json").write_text(json.dumps({"frames": frames}),
                                     encoding="utf-8")
    client.post("/api/golden/import",
                json={"source_dir": str(src), "name": "ctset",
                      "image_format": "keep"})
    info = client.get("/api/golden/ctset").json()
    for entry in info["entries"]:
        client.post("/api/golden/ctset/entry",
                    json={"photo": entry["photo"], "content": content})
    # train the template bank so backend=bank is usable without torch
    assert glyphs.build()["cells"] == 24
    classifier.train_bank()

    # single read: read string matches the curated content exactly
    photo = info["entries"][0]["photo"]
    r = client.post("/api/cnn-test/read",
                    json={"set": "ctset", "photo": photo,
                          "backend": "bank"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["read"] == content
    assert body["mismatches"] == 0
    assert len(body["modules"]) == 12
    assert body["model"]["backend"] == "bank"
    assert body["detected"] is True

    # full benchmark over the set
    r = client.post("/api/cnn-test/start",
                    json={"set": "ctset", "backend": "bank"})
    assert r.status_code == 200, r.text
    deadline = time.time() + 15
    while time.time() < deadline:
        st = server.cnn_test.state()
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    st = server.cnn_test.state()
    assert st["status"] == "done", st
    assert st["report"]["summary"]["vlm"]["mean"] == 0.0
    assert st["report"]["dataset"]["set"] == "ctset"
    # the bank was trained on this set: the run must flag itself in-sample
    assert st["report"]["dataset"]["insample"] is True
    assert st["report"]["pipeline"]["text"].startswith("read=classify")
    rows = client.get("/api/cnn-test/rows").json()["rows"]
    assert len(rows) == 2 and all(r_["mm_vlm"] == 0 for r_ in rows)
    detail = client.get("/api/cnn-test/row/0").json()
    assert len(detail["modules"]) == 12
    runs = client.get("/api/cnn-test/runs").json()["runs"]
    assert runs and runs[0]["set"] == "ctset"
    assert runs[0]["insample"] is True


# -- calibration -----------------------------------------------------------------

def test_cal_start_validation(client):
    r = client.post("/api/cal/start", json={"approach": "bogus"})
    assert r.status_code == 400
    r = client.post("/api/cal/start", json={"approach": "cnn"})
    assert r.status_code == 400
    assert "local classifier not usable" in r.json()["detail"]
    r = client.post("/api/cal/start",
                    json={"approach": "vlm", "phases": ["p1", "nope"]})
    assert r.status_code == 400
    assert "unknown phases" in r.json()["detail"]
    # Default provider (opencode.ai) has a base URL + model, so the guard
    # that fires is the API key.
    r = client.post("/api/cal/start", json={"approach": "vlm"})
    assert r.status_code == 400
    assert "llm_api_key" in r.json()["detail"]


def test_cal_photo_guard(client):
    assert client.get("/api/cal/photo/nope.png").status_code == 404
    assert client.get("/api/cal/photo/..%5Cx.png").status_code == 400


def test_restore_snapshot_rejected_while_run_is_active(client, tmp_path,
                                                       monkeypatch):
    """A mid-run rollback must be refused (it would desync the calibrator).

    Restoring while the calibrator is live reverts committed offsets
    underneath it, and the calibrator keeps tracking its own
    overlay/residue belief: the next preview commit would then persist from
    a stale base and write a wrong absolute offset to NVS.
    """
    calls: list[str] = []

    class RecordingDisplay:
        def __init__(self, host):
            calls.append(host)

        def restore(self, snapshot):
            raise AssertionError("restore must not run while a run is active")

        def reload(self):
            raise AssertionError("reload must not run while a run is active")

    monkeypatch.setattr(server, "Display", RecordingDisplay)
    with server.cal.lock:
        server.cal.run_dir = str(tmp_path)
    for status in ("running", "aborting"):
        with server.cal.lock:
            server.cal.status = status
        r = client.post("/api/cal/restore-snapshot")
        assert r.status_code == 409, r.text
    assert calls == []  # the display was never contacted


def test_restore_snapshot_works_when_idle(client, tmp_path, monkeypatch):
    snapshot = {"settings": {"moduleOffsets": [1, 2]}}
    run_dir = tmp_path / "cal-002"
    run_dir.mkdir()
    (run_dir / "snapshot.json").write_text(json.dumps(snapshot),
                                           encoding="utf-8")
    restored: list = []

    class RecordingDisplay:
        def __init__(self, host):
            pass

        def restore(self, snap):
            restored.append(snap)
            return {"type": "success"}

        def reload(self):
            return {"type": "success"}

    monkeypatch.setattr(server, "Display", RecordingDisplay)
    with server.cal.lock:
        server.cal.run_dir = str(run_dir)
        server.cal.status = "done"
    r = client.post("/api/cal/restore-snapshot")
    assert r.status_code == 200, r.text
    assert restored == [snapshot]


def test_start_does_not_wedge_status_when_run_dir_allocation_fails(
        client, monkeypatch):
    """A failing run-dir allocation must not leave a harness "running".

    ``alloc_run_dir`` touches the disk, so it can fail (full disk,
    unwritable ``CALIB_AUTO_DATA``). It used to run AFTER the status flip
    and outside any handler, so the exception escaped ``start()`` with the
    status stuck at "running" and no thread behind it: every later start
    409'd until a server restart, and ``abort()`` only moves the status to
    "aborting", which is guarded too.
    """
    real_alloc = server.paths.alloc_run_dir

    def boom(_prefix="run"):
        raise OSError("disk full")

    monkeypatch.setattr(server.paths, "alloc_run_dir", boom)
    cfg = server.config_mod.load_config()
    cfg["llm_api_key"] = "sk-test"  # satisfy the VLM recognizer guard

    harnesses = (
        (server.cal, lambda: server.cal.start(cfg, {"approach": "vlm"})),
        (server.bench, lambda: server.bench.start(cfg)),
        (server.train_job, lambda: server.train_job.start({})),
    )
    for harness, call in harnesses:
        with harness.lock:
            harness.status = "idle"
        with pytest.raises(server.HTTPException) as excinfo:
            call()
        assert excinfo.value.status_code == 500
        assert "cannot allocate a run directory" in str(excinfo.value.detail)
        with harness.lock:
            assert harness.status == "idle", harness.status
            assert harness.run_dir in ("", None), harness.run_dir

    # Not wedged: with a working allocator the same harness starts again.
    monkeypatch.setattr(server.paths, "alloc_run_dir", real_alloc)
    server.bench.start(cfg)
    assert server.bench.state()["run_dir"]


def test_train_job_bad_epochs_does_not_wedge_status(client):
    """A malformed epochs value must 400, not leave the job stuck running."""
    with server.train_job.lock:
        server.train_job.status = "idle"
    with pytest.raises(server.HTTPException) as excinfo:
        server.train_job.start({"epochs": "not-a-number"})
    assert excinfo.value.status_code == 400
    with server.train_job.lock:
        assert server.train_job.status == "idle"


# -- training job ----------------------------------------------------------------

def test_train_job_runs_with_stubbed_trainer(client, tmp_path, monkeypatch):
    src = _make_source(tmp_path)
    client.post("/api/golden/import",
                json={"source_dir": str(src), "name": "t1",
                      "image_format": "keep"})
    # training consumes verified entries only: verify both
    info = client.get("/api/golden/t1").json()
    for entry in info["entries"]:
        client.post("/api/golden/t1/entry",
                    json={"photo": entry["photo"], "content": "A" * 12})

    seen: dict = {}

    def fake_train(**kwargs):
        seen["sets"] = list(kwargs.get("sets") or [])
        on_epoch = kwargs.get("on_epoch")
        if on_epoch:
            on_epoch({"epoch": 1, "loss": 0.5, "val_acc": 0.9})
        return {"model": object(), "classes": ["A"], "size": 64,
                "metrics": {"cnn_acc": 0.9, "rows": 1, "row_exact": 1,
                            "row_mismatch_avg": 0.0, "test_cells": 1,
                            "test_photos": 1, "per_class": {},
                            "confusions": []},
                "excluded": {}, "history": [],
                "sets": list(kwargs.get("sets") or []),
                "cells": 24, "train_cells": 20}

    monkeypatch.setattr(train_cnn, "train", fake_train)
    monkeypatch.setattr(train_cnn, "save_model",
                        lambda result, path, meta=None: path)

    # Selecting a subset of golden sets must flow into the training call.
    r = client.post("/api/train/start", json={"epochs": 1,
                                               "sets": ["t1"]})
    assert r.status_code == 200, r.text
    deadline = time.time() + 10
    while time.time() < deadline:
        st = server.train_job.state()
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.05)
    st = server.train_job.state()
    assert st["status"] == "done", st
    assert seen["sets"] == ["t1"]
    assert st["result"]["cells"] == 24
    assert st["result"]["sets"] == ["t1"]
    assert st["result"]["metrics"]["cnn_acc"] == 0.9
    texts = [e["text"] for e in st["events"]]
    assert any("epoch" in t for t in texts)
    assert any("training on 1 set(s): t1" in t for t in texts)
    # Per-epoch records accumulate in state for the live loss graph.
    assert st["history"] == [{"epoch": 1, "loss": 0.5, "val_acc": 0.9}]

    # The training history records which sets each model was built on.
    runs = client.get("/api/train/runs").json()["runs"]
    assert runs and runs[0]["sets"] == ["t1"]
    assert runs[0]["finished"]
    assert runs[0]["val_acc"] == 0.9


# -- live view / exposure -------------------------------------------------------

def test_exposure_resolution_prefers_request():
    cfg = {"exposure": -7.0}
    # No request key -> the stored manual value is used.
    assert server._exposure_of({}, cfg) == -7.0
    assert server._exposure_of(None, cfg) == -7.0
    # An explicit request always wins, including "auto" (= driver AE).
    assert server._exposure_of({"exposure": -3}, cfg) == -3.0
    assert server._exposure_of({"exposure": None}, cfg) is None
    assert server._exposure_of({"exposure": ""}, cfg) is None
    assert server._exposure_of({"exposure": "auto"}, cfg) is None
    assert server._exposure_of({"exposure": "AUTO"}, cfg) is None
    assert server._exposure_of({"exposure": "nonsense"}, cfg) is None


def test_camera_frame_accepts_auto_exposure(client):
    """`exposure=auto` must not be a request-validation error.

    Without a camera the call fails later with a 400 camera error, so any
    422 here means the query param is still typed as a number.
    """
    r = client.get("/api/camera/frame?exposure=auto")
    assert r.status_code != 422, r.text
    assert client.get("/api/camera/frame").status_code != 422


def test_camera_frame_rejects_bad_exposure(client):
    r = client.get("/api/camera/frame?exposure=bogus")
    assert r.status_code == 400, r.text
    assert "number or 'auto'" in r.json()["detail"]
    r = client.get("/api/camera/frame?warmup_s=bogus")
    assert r.status_code == 400, r.text
    assert "warmup_s must be a number" in r.json()["detail"]
