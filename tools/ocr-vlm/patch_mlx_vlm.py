"""Patch an installed mlx-vlm with the cross-thread MLX stream workaround.

Why: the mlx-vlm OpenAI server preprocesses multimodal inputs on the request
thread (``asyncio.to_thread``) but runs generation on a dedicated GPU worker
thread.  DeepSeek-OCR-2's processor builds its image tensors with lazy MLX ops
on the request thread, and the model force-evaluates them on the GPU thread —
MLX streams are thread-local, so every request fails with::

    RuntimeError: There is no Stream(gpu, N) in current thread.

This patch eager-evaluates (``mx.eval``) the preprocessed inputs right after
preprocessing, so no lazily-scheduled graph from the request thread is ever
finalized on the GPU thread.  It is the same fix pattern upstream applied for
its own cross-thread bugs (mlx-vlm issues #1591/#1614, PR #1854) and can be
removed once the fix lands in a release.

Usage (run on the machine that hosts the mlx-vlm server, inside its env)::

    python patch_mlx_vlm.py            # apply (idempotent, makes a .bak once)
    python patch_mlx_vlm.py --check    # report status, change nothing
    python patch_mlx_vlm.py --revert   # remove the patch again
    python patch_mlx_vlm.py --file /path/to/generation.py   # explicit target

Restart the mlx-vlm server after applying.  Upgrading or reinstalling mlx-vlm
removes the patch (re-run this script if the bug is still present).
"""

from __future__ import annotations

import argparse
from pathlib import Path

MARKER = "PATCH-OCR-VLM"
ANCHOR = "raw_inputs = self._preprocess_request"
BLOCK = [
    "# PATCH-OCR-VLM: eager-eval preprocessed inputs so the GPU worker thread",
    "# never finalizes graphs created on this request thread (MLX streams are",
    '# thread-local; fixes "There is no Stream(gpu, N) in current thread").',
    "import mlx.core as _mx",
    "from mlx.utils import tree_flatten as _tf",
    "_arrays = [v for _, v in _tf(raw_inputs) if isinstance(v, _mx.array)]",
    "if _arrays:",
    "    _mx.eval(*_arrays)",
]


def resolve_target(explicit: str | None) -> Path:
    """The file to patch: --file, or the installed mlx_vlm generation module."""
    if explicit:
        return Path(explicit)
    try:
        import mlx_vlm.server.generation as gen
    except Exception as exc:  # noqa: BLE001 - report any import problem clearly
        raise SystemExit(
            f"cannot import mlx_vlm.server.generation ({exc});\n"
            "run this script inside the environment that hosts the server, "
            "or pass --file PATH to the file to patch"
        )
    return Path(gen.__file__)


def apply_patch(text: str) -> tuple[str, int]:
    """Insert the eager-eval block after every preprocessing assignment."""
    out: list[str] = []
    hits = 0
    for line in text.splitlines(keepends=True):
        out.append(line)
        if ANCHOR in line:
            indent = line[: len(line) - len(line.lstrip())]
            out.extend(indent + block_line + "\n" for block_line in BLOCK)
            hits += 1
    return "".join(out), hits


def revert_patch(text: str) -> tuple[str, int]:
    """Remove previously inserted blocks (refuses on unexpected content)."""
    lines = text.splitlines(keepends=True)
    expected = [line.strip() for line in BLOCK]
    out: list[str] = []
    hits = 0
    index = 0
    while index < len(lines):
        if lines[index].strip().startswith("# " + MARKER):
            segment = [line.strip() for line in lines[index:index + len(BLOCK)]]
            if segment != expected:
                raise SystemExit(
                    f"line {index + 1}: marked block does not match the known "
                    "patch content; refusing to revert (edit the file manually)"
                )
            index += len(BLOCK)
            hits += 1
            continue
        out.append(lines[index])
        index += 1
    return "".join(out), hits


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", default=None,
                        help="target file (default: installed "
                             "mlx_vlm/server/generation.py)")
    parser.add_argument("--check", action="store_true",
                        help="report status only, change nothing")
    parser.add_argument("--revert", action="store_true",
                        help="remove the patch again")
    args = parser.parse_args()

    path = resolve_target(args.file)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cannot read {path}: {exc}")
        return 2

    patched = MARKER in text
    anchors = sum(1 for line in text.splitlines() if ANCHOR in line)

    if args.check:
        print(f"file: {path}")
        print(f"patched: {'yes' if patched else 'no'}")
        print(f"anchor lines: {anchors}")
        return 0

    if args.revert:
        if not patched:
            print(f"nothing to revert in {path} (marker not present)")
            return 0
        new_text, hits = revert_patch(text)
        path.write_text(new_text, encoding="utf-8")
        print(f"reverted {hits} site(s) in {path}")
        return 0

    if patched:
        print(f"already patched: {path}")
        return 0
    if not anchors:
        print(f"ANCHOR NOT FOUND in {path}")
        print(f"inspect the file yourself:  grep -n preprocess {path}")
        return 1

    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"backup written: {backup}")

    new_text, hits = apply_patch(text)
    path.write_text(new_text, encoding="utf-8")
    print(f"patched {hits} site(s) in {path}")
    print("now restart the mlx-vlm server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
