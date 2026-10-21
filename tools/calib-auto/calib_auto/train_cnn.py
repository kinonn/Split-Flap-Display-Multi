"""Local glyph classifier: CNN backend.

The template bank proved the approach — geometry-normalized glyph
rasters classify at ~95% reader-simulated accuracy with leave-one-set-out
validation — but its per-class pixel means cannot absorb the hard
captures: motion-ghosted flaps, tilted mid-transition glyphs, specular
lip glints, and the genuinely confusable pairs (``S``/``5``, ``6``/``H``,
and glint texture vs the dash). A small CNN trained on the same curated
crops learns those invariances, plus the ``space`` class itself
(blank-flap texture included), instead of relying on a hand-tuned ink
threshold for blank gating.

Design constraints, all deliberate:

- **Same canonical rasters as the bank** (``segment.canonical_glyph`` at
  ``GLYPH_SIZE``): no separate preprocessing pipeline, and evaluation
  is comparable cell-for-cell.
- **Photo-disjoint validation.** Photos are split by a deterministic
  hash of ``set/photo``; the model is *never* validated on a photo it
  trained on. ``all_data=True`` trains the deployment artifact on
  everything after the split run reports honest numbers.
- **Augmentation is plain OpenCV/numpy** (affine, brightness/contrast,
  blur, noise, erase): no torchvision dependency, and the transforms
  mirror the capture defects actually present in the data.
- **Verified labels only**, curated char set only (same filter and
  exclusion counting as the bank).

CLI::

    python -m calib_auto.train_cnn [--epochs 30] [--size 64]
                                   [--batch 128] [--lr 1e-3]
                                   [--photo-frac 0.2] [--seed 1]
                                   [--device auto|cpu|cuda|mps]

Training uses CUDA when available, else MPS when available, else CPU
(``--device`` or ``CALIB_AUTO_DEVICE`` pins a backend). Model and batch
tensors live on that device; data prep (OpenCV/numpy augmentation)
stays on CPU. Checkpoints are stored CPU-side so the ``cnn.pt``
artifact loads anywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import zlib
from datetime import datetime, timezone

import cv2
import numpy as np

from . import classifier, glyphs, paths

MODEL_FILE = "cnn.pt"
MODEL_META = "cnn.json"
DEFAULT_SIZE = glyphs.GLYPH_SIZE


# -- device --------------------------------------------------------------------

def resolve_device(preferred: str | None = None):
    """Pick the torch device for training/inference.

    Order: an explicit ``preferred`` (or the ``CALIB_AUTO_DEVICE`` env
    var) when it names a usable device, else CUDA when available, else
    MPS when available, else CPU. An unavailable explicit choice falls
    back through the same auto order so training never fails hard just
    because a GPU went away.
    """
    import torch

    raw = str(preferred if preferred is not None
              else os.environ.get("CALIB_AUTO_DEVICE") or "auto")
    raw = raw.strip().lower() or "auto"
    if raw in ("auto", ""):
        if torch.cuda.is_available():
            return torch.device("cuda")
        try:
            if torch.backends.mps.is_available():
                return torch.device("mps")
        except AttributeError:
            pass
        return torch.device("cpu")
    try:
        dev = torch.device(raw)
    except (RuntimeError, ValueError):
        dev = None
    if dev is not None:
        if dev.type == "cuda" and torch.cuda.is_available():
            return dev
        if dev.type == "mps":
            try:
                if torch.backends.mps.is_available():
                    return dev
            except AttributeError:
                pass
        if dev.type == "cpu":
            return dev
    # Explicit choice unusable: fall back through the auto order.
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")


def _model_device(model):
    """Device a (possibly freshly built) model lives on; CPU fallback."""
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError):
        import torch

        return torch.device("cpu")


# -- data ----------------------------------------------------------------------

class GlyphData:
    """Cache rasters + labels, split by a photo-disjoint hash rule."""

    def __init__(self, caches: list[dict], *,
                 charset: str | None = None):
        allowed = set(charset) if charset is not None else set(
            classifier.CHARSET)
        rasters: list[np.ndarray] = []
        labels: list[str] = []
        keys: list[str] = []
        positions: list[int] = []
        self.excluded: dict[str, int] = {}
        for payload in caches:
            for i in range(len(payload["labels"])):
                label = str(payload["labels"][i])
                if label not in allowed:
                    self.excluded[label] = self.excluded.get(label, 0) + 1
                    continue
                rasters.append(payload["crops"][i])
                labels.append(label)
                keys.append(f"{payload['set']}/{payload['photos'][i]}")
                positions.append(int(payload["positions"][i]))
        self.rasters = (np.stack(rasters) if rasters
                        else np.zeros((0, DEFAULT_SIZE, DEFAULT_SIZE),
                                      np.uint8))
        self.labels = np.array(labels, dtype=object).astype(str)
        self.keys = keys
        self.positions = positions
        self.classes = sorted(set(labels))
        self.class_index = {c: i for i, c in enumerate(self.classes)}
        self.targets = np.array([self.class_index[c] for c in labels],
                                dtype=np.int64)

    def test_mask(self, frac: float) -> np.ndarray:
        cutoff = round(max(0.0, min(0.9, frac)) * 100)
        return np.array([zlib.crc32(k.encode()) % 100 < cutoff
                         for k in self.keys], dtype=bool)


# -- model ---------------------------------------------------------------------

def _conv_block(cin: int, cout: int) -> object:
    from torch import nn

    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


def build_model(n_classes: int, ch: tuple[int, int, int] = (16, 32, 64),
                input_size: int = DEFAULT_SIZE):
    from torch import nn

    # three 2x pools: spatial side = input_size // 8
    spatial = max(1, int(input_size) // 8)
    return nn.Sequential(
        _conv_block(1, ch[0]),
        _conv_block(ch[0], ch[1]),
        _conv_block(ch[1], ch[2]),
        nn.Flatten(),
        nn.Linear(ch[2] * spatial * spatial, 192), nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(192, n_classes),
    )


# -- tensor plumbing -----------------------------------------------------------

def _standardize(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    return (img - img.mean()) / (img.std() + 1e-6)


def _to_tensor(batch: np.ndarray, device=None):
    import torch

    tensor = torch.from_numpy(batch.astype(np.float32)).unsqueeze(1)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _prep_batch(rasters: np.ndarray, size: int) -> np.ndarray:
    out = np.empty((len(rasters), size, size), np.float32)
    for i, raster in enumerate(rasters):
        img = raster
        if img.shape[:2] != (size, size):
            img = cv2.resize(img, (size, size),
                             interpolation=cv2.INTER_AREA)
        out[i] = _standardize(img) * 0.25
    return out


def augment(raster: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One training sample: affine + photometric jitter + erase.

    The transforms mirror the dataset's real defects (flap tilt and
    scale during capture, specular hotspots, motion blur, sensor noise)
    rather than generic ImageNet recipes.
    """
    size = raster.shape[0]
    img = raster.astype(np.float32)
    angle = float(rng.uniform(-7.0, 7.0))
    scale = float(rng.uniform(0.90, 1.10))
    mat = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5),
                                  angle, scale)
    mat[0, 2] += float(rng.uniform(-0.08, 0.08)) * size
    mat[1, 2] += float(rng.uniform(-0.08, 0.08)) * size
    img = cv2.warpAffine(img, mat, (size, size),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    img = img * float(rng.uniform(0.80, 1.20)) + float(rng.uniform(-25, 25))
    if rng.random() < 0.5:
        sigma = float(rng.uniform(0.3, 1.2))
        img = cv2.GaussianBlur(img, (3, 3), sigma)
    if rng.random() < 0.5:
        img = img + rng.normal(0.0, float(rng.uniform(2.0, 10.0)),
                               img.shape).astype(np.float32)
    if rng.random() < 0.25:
        w = int(rng.uniform(0.08, 0.20) * size)
        h = int(rng.uniform(0.08, 0.20) * size)
        x = int(rng.integers(0, max(1, size - w)))
        y = int(rng.integers(0, max(1, size - h)))
        img[y:y + h, x:x + w] = float(rng.uniform(0, 255))
    return np.clip(img, 0, 255).astype(np.uint8)


# -- training ------------------------------------------------------------------

def train(*, sets: list[str] | None = None,
          epochs: int = 30, size: int = DEFAULT_SIZE, batch: int = 128,
          lr: float = 1e-3, photo_frac: float = 0.2, seed: int = 1,
          all_data: bool = False, verbose: bool = True,
          on_epoch=None, device: str | None = None) -> dict:
    """Train and return {model, classes, metrics, history, ...}.

    ``all_data=False``: photo-disjoint split; metrics are the honest
    generalization numbers. ``all_data=True``: train on everything (the
    deployment artifact). ``on_epoch(record)`` streams per-epoch
    progress to the training UI. ``device`` selects the torch device:
    ``None``/``"auto"`` (the default) uses CUDA when available, else
    MPS when available, else CPU; ``"cpu"``, ``"cuda"`` (or
    ``"cuda:N"``) and ``"mps"`` pin a backend, falling back through the
    same order when the pinned backend is unavailable. The
    ``CALIB_AUTO_DEVICE`` env var plays the same role as ``device``.
    """
    import torch
    from torch import nn

    device_obj = resolve_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device_obj.type == "cuda":
        try:
            torch.cuda.manual_seed_all(seed)
        except (AttributeError, RuntimeError):
            pass
    else:
        # Thread cap is a CPU-only tuning knob; accelerators manage
        # their own parallelism.
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    caches = classifier.load_caches(sets)
    if not caches:
        raise classifier.BankError(
            "no glyph caches found — run: python -m calib_auto.glyphs")
    gd = GlyphData(caches)
    if len(gd.classes) < 2:
        raise classifier.BankError("need at least two classes to train")

    test = gd.test_mask(photo_frac) if not all_data else np.zeros(
        len(gd.labels), dtype=bool)
    train_idx = np.flatnonzero(~test)
    test_idx = np.flatnonzero(test)
    if all_data:
        train_idx = np.arange(len(gd.labels))

    model = build_model(len(gd.classes), input_size=size).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=1e-4)
    steps = max(1, int(np.ceil(len(train_idx) / batch))) * max(1, epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    counts = np.bincount(gd.targets[train_idx],
                         minlength=len(gd.classes)).astype(np.float32)
    weights = np.sqrt(len(train_idx) / (len(gd.classes)
                                        * np.maximum(counts, 1.0)))
    weights = weights / weights.mean()
    crit = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).to(device_obj),
        label_smoothing=0.05)

    rng = np.random.default_rng(seed)
    best_state = None
    best_acc = -1.0
    history: list[dict] = []
    if verbose:
        print(f"  device: {device_obj}")
    for epoch in range(max(1, epochs)):
        model.train()
        order = rng.permutation(train_idx)
        total_loss = 0.0
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            x = _prep_batch(np.stack([augment(gd.rasters[i], rng)
                                      for i in idx]), size)
            xb = _to_tensor(x, device_obj)
            y = torch.from_numpy(gd.targets[idx]).to(device_obj)
            optimizer.zero_grad()
            loss = crit(model(xb), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach()) * len(idx)
        rec = {"epoch": epoch + 1, "loss": round(total_loss / len(order), 4)}
        if not all_data and len(test_idx):
            acc = _accuracy(model, gd, test_idx, size, batch)
            rec["val_acc"] = round(acc, 4)
            if acc > best_acc:
                best_acc = acc
                # Keep the checkpoint on CPU so the artifact (and the
                # in-memory model handed to the caller) loads anywhere.
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
        history.append(rec)
        if verbose:
            extra = (f" val {rec['val_acc']:.4f}"
                     if "val_acc" in rec else "")
            print(f"  epoch {rec['epoch']:2d}: loss {rec['loss']:.4f}{extra}")
        if on_epoch:
            on_epoch(rec)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    metrics = None
    if not all_data and len(test_idx):
        metrics = evaluate_model(model, gd, test_idx, size, batch)
        metrics["photo_frac"] = photo_frac
        metrics["seed"] = seed
        metrics["device"] = str(device_obj)
    # Per-class cells the model actually trained on (train split, or all
    # cells for the deployment artifact) — persisted into cnn.json so the
    # training UI can show data-vs-model coverage per character.
    seen = np.bincount(gd.targets[train_idx], minlength=len(gd.classes))
    class_counts = {cls: int(seen[i]) for i, cls in enumerate(gd.classes)}
    return {"model": model, "classes": gd.classes, "metrics": metrics,
            "size": size, "excluded": gd.excluded,
            "history": history, "sets": sorted({c["set"] for c in caches}),
            "cells": len(gd.labels), "train_cells": len(train_idx),
            "device": str(device_obj), "class_counts": class_counts}


def _predict_idx(model, rasters: np.ndarray, size: int, batch: int):
    import torch

    device = _model_device(model)
    out = np.empty(len(rasters), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(rasters), batch):
            chunk = rasters[start:start + batch]
            x = _prep_batch(chunk, size)
            out[start:start + len(chunk)] = (
                model(_to_tensor(x, device)).argmax(dim=1).cpu().numpy())
    return out


def _accuracy(model, gd: GlyphData, idx: np.ndarray, size: int,
              batch: int) -> float:
    preds = _predict_idx(model, gd.rasters[idx], size, batch)
    return float(np.mean(preds == gd.targets[idx]))


def evaluate_model(model, gd: GlyphData, test_idx: np.ndarray, size: int,
                   batch: int) -> dict:
    """Same metric vocabulary as the bank's ``evaluate`` (sim columns)."""
    import torch

    probs_fn = torch.nn.Softmax(dim=1)
    device = _model_device(model)
    preds = np.empty(len(test_idx), dtype=np.int64)
    confs = np.empty(len(test_idx), dtype=np.float32)
    margins = np.empty(len(test_idx), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(test_idx), batch):
            idx = test_idx[start:start + batch]
            x = _prep_batch(gd.rasters[idx], size)
            probs = probs_fn(model(_to_tensor(x, device))).cpu().numpy()
            order = np.argsort(probs, axis=1)[:, ::-1]
            preds[start:start + len(idx)] = order[:, 0]
            confs[start:start + len(idx)] = probs[np.arange(len(idx)),
                                                  order[:, 0]]
            if order.shape[1] > 1:
                margins[start:start + len(idx)] = (
                    probs[np.arange(len(idx)), order[:, 0]]
                    - probs[np.arange(len(idx)), order[:, 1]])
            else:
                margins[start:start + len(idx)] = 1.0

    truth_idx = gd.targets[test_idx]
    acc = float(np.mean(preds == truth_idx))
    rows: dict[str, list[tuple[int, int, int]]] = {}
    for out_i, cell in enumerate(test_idx):
        rows.setdefault(gd.keys[cell], []).append(
            (gd.positions[cell], int(preds[out_i]), int(truth_idx[out_i])))
    exact = 0
    mismatches = 0
    for cells in rows.values():
        cells.sort()
        mm = sum(1 for _, got, want in cells if got != want)
        mismatches += mm
        if mm == 0:
            exact += 1

    per_class: dict[str, dict] = {}
    confusions: dict[tuple[str, str], int] = {}
    for out_i, cell in enumerate(test_idx):
        want = gd.classes[int(truth_idx[out_i])]
        got = gd.classes[int(preds[out_i])]
        entry = per_class.setdefault(want, {"total": 0, "correct": 0})
        entry["total"] += 1
        if want == got:
            entry["correct"] += 1
        else:
            confusions[(want, got)] = confusions.get((want, got), 0) + 1
    top = sorted(confusions.items(), key=lambda kv: -kv[1])[:12]
    low = confs < 0.9
    return {
        "split": "photo",
        "size": size,
        "test_cells": len(test_idx),
        "test_photos": len(rows),
        "cnn_acc": round(acc, 4),
        "rows": len(rows),
        "row_exact": exact,
        "row_exact_rate": round(exact / len(rows), 4) if rows else 0.0,
        "row_mismatch_avg": (round(mismatches / len(rows), 4)
                             if rows else 0.0),
        "low_conf_cells": int(low.sum()),
        "low_conf_frac": round(float(low.mean()), 4) if len(low) else 0.0,
        "per_class": {k: {**v, "recall": round(v["correct"] / v["total"], 4)}
                      for k, v in sorted(per_class.items())},
        "confusions": [{"label": a, "pred": b, "count": n}
                       for (a, b), n in top],
    }


def print_metrics(metrics: dict) -> None:
    print(f"  per-cell cnn acc: {metrics['cnn_acc']:.4f} over "
          f"{metrics['test_cells']} cells / {metrics['test_photos']} photos")
    print(f"  rows: {metrics['row_exact']}/{metrics['rows']} exact "
          f"({metrics['row_exact_rate']:.3f}) · avg mismatches "
          f"{metrics['row_mismatch_avg']}")
    weak = [(c, v) for c, v in metrics["per_class"].items()
            if v["recall"] < 1.0]
    if weak:
        shown = ", ".join(f"{c!r} {v['recall']:.3f} ({v['correct']}/"
                          f"{v['total']})" for c, v in weak[:14])
        print(f"  classes below 1.0 recall: {shown}")
    if metrics["confusions"]:
        top = ", ".join(f"{c['label']!r}->{c['pred']!r} {c['count']}"
                        for c in metrics["confusions"][:8])
        print(f"  confusions: {top}")


# -- deployment wrapper --------------------------------------------------------

class CnnClassifier:
    """Same prediction interface as ``GlyphBank`` (reader duck-typing)."""

    def __init__(self, model, classes: list[str], size: int,
                 device: str | None = None):
        self.model = model
        self.classes = list(classes)
        self.size = int(size)
        self.meta: dict = {}
        if device is not None:
            dev = resolve_device(device)
            try:
                self.model = self.model.to(dev)
            except AttributeError:
                pass
            self.device = dev
        else:
            try:
                self.device = _model_device(self.model)
            except (NameError, RuntimeError):
                import torch

                self.device = torch.device("cpu")

    @classmethod
    def load(cls, path: str, device: str | None = None) -> CnnClassifier:
        import torch

        if not os.path.isfile(path):
            raise classifier.BankError(f"no CNN model at {path}")
        try:
            payload = torch.load(path, map_location="cpu",
                                 weights_only=False)
        except Exception as exc:  # corrupt/old artifact
            raise classifier.BankError(
                f"cannot load CNN model at {path}: {exc}") from exc
        model = build_model(len(payload["classes"]),
                            tuple(payload.get("ch", (16, 32, 64))),
                            int(payload.get("input_size",
                                            payload.get("size",
                                                        DEFAULT_SIZE))))
        model.load_state_dict(payload["state_dict"])
        dev = resolve_device(device)
        model = model.to(dev)
        model.eval()
        wrapper = cls(model, payload["classes"], payload.get("size",
                                                             DEFAULT_SIZE))
        wrapper.device = dev
        meta_file = os.path.splitext(path)[0] + ".json"
        if os.path.isfile(meta_file):
            try:
                with open(meta_file, encoding="utf-8") as fh:
                    wrapper.meta = json.load(fh)
            except (OSError, ValueError):
                wrapper.meta = {}
        return wrapper

    def predict_raster(self, raster) -> tuple[str, float, float] | None:
        import torch

        if raster is None or getattr(raster, "size", 0) == 0:
            return None
        dev = getattr(self, "device", None)
        if dev is None:
            dev = _model_device(self.model)
        x = _prep_batch(np.stack([raster]), self.size)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(
                self.model(_to_tensor(x, dev)), dim=1)[0].cpu().numpy()
        order = np.argsort(probs)[::-1]
        best = int(order[0])
        conf = float(probs[best])
        second = float(probs[int(order[1])]) if len(order) > 1 else 0.0
        return self.classes[best], conf, conf - second


def save_model(result: dict, path: str, *, meta: dict | None = None) -> str:
    import torch

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    # Keep the artifact portable: a GPU-trained checkpoint must still
    # load on a CPU-only machine (load uses map_location="cpu").
    state = {k: v.detach().cpu().clone()
             for k, v in result["model"].state_dict().items()}
    payload = {
        "state_dict": state,
        "classes": result["classes"],
        "size": result["size"],
        "input_size": result["size"],
        "ch": [16, 32, 64],
    }
    torch.save(payload, path)
    doc = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": "cnn",
        "size": result["size"],
        "classes": result["classes"],
        "cells": result["cells"],
        "train_cells": result["train_cells"],
        "sets": result["sets"],
        "excluded_labels": result["excluded"],
        "charset": classifier.CHARSET,
        "device": result.get("device", "cpu"),
        "class_counts": result.get("class_counts", {}),
    }
    doc.update(meta or {})
    tmp = os.path.splitext(path)[0] + ".json.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, os.path.splitext(path)[0] + ".json")
    return path


def model_path() -> str:
    return os.path.join(paths.models_root(), MODEL_FILE)


# -- CLI -----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="calib_auto.train_cnn",
        description="Train the CNN glyph backend on the golden glyph caches")
    parser.add_argument("--sets", nargs="*", default=None,
                        help="only these glyph-cache sets (default: all)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--photo-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None,
                        help="torch device: auto (default: CUDA > MPS > CPU), "
                             "cpu, cuda, cuda:N or mps "
                             "(env CALIB_AUTO_DEVICE overrides the default)")
    parser.add_argument("--train-all-only", action="store_true",
                        help="skip the photo-split run (no honest metrics)")
    parser.add_argument("--no-final", action="store_true",
                        help="do not retrain on all data; keep split model")
    args = parser.parse_args(argv)

    if not args.train_all_only:
        print(f"photo-split run (val {int(args.photo_frac * 100)}%, "
              f"{args.epochs} epochs)")
        result = train(sets=args.sets, epochs=args.epochs,
                       size=args.size, batch=args.batch, lr=args.lr,
                       photo_frac=args.photo_frac, seed=args.seed,
                       device=args.device)
        metrics = result["metrics"]
        if metrics:
            print_metrics(metrics)
        excluded = result["excluded"]
        if excluded:
            print("  excluded labels (not in charset): "
                  + ", ".join(f"{c!r} x{n}" for c, n in excluded.items()))
        if args.no_final:
            path = save_model(result, model_path(),
                              meta={"metrics": metrics,
                                    "trained": "photo-split"})
            print(f"saved (split model): {path}")
            return 0
        print("final run on all golden cells (deployment artifact)")
        final = train(sets=args.sets, epochs=args.epochs,
                      size=args.size, batch=args.batch, lr=args.lr,
                      seed=args.seed, all_data=True, verbose=False,
                      device=args.device)
        path = save_model(final, model_path(),
                          meta={"metrics": metrics, "trained": "all-data"})
        print(f"saved: {path}")
        return 0
    final = train(sets=args.sets, epochs=args.epochs,
                  size=args.size, batch=args.batch, lr=args.lr,
                  seed=args.seed, all_data=True, verbose=True,
                  device=args.device)
    path = save_model(final, model_path(),
                      meta={"metrics": None, "trained": "all-data"})
    print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
