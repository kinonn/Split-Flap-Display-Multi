"""splitflap-calib: one-command auto-calibration.

Usage:
    uv run splitflap-calib --host splitflap.local --photo-dir ./photos
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .camera import Camera, CameraError
from .display import CalibError, Display
from .loop import SUPPORTED_CONTRACT, Calibrator, load_bundled_contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vision-guided auto-calibration for the split-flap display.")
    parser.add_argument("--host", required=True,
                        help="Master hostname/IP, e.g. splitflap.local")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--photo-dir", default="./calib-photos")
    parser.add_argument("--phase", type=int, choices=(1, 2, 3, 4), default=4,
                        help="1=read-only proposals, 2=+volatile previews, "
                             "3=+persist, 4=+acceptance (default)")
    parser.add_argument("--dwell-ms", type=int, default=800)
    parser.add_argument("--timeout-s", type=float, default=60.0,
                        help="Settle deadline per move (s); connection fails fast")
    parser.add_argument("--identity-thresh", type=float, default=0.85,
                        help="Peer-similarity below which a clean crop is flagged "
                             "as the wrong glyph (0..1)")
    parser.add_argument("--relearn-templates", action="store_true",
                        help="Ignore stored golden templates and rebuild the "
                             "glyph bank from this run")
    parser.add_argument("--contract", default=None,
                        help="Override bundled calib/contract.json")
    parser.add_argument("--check-camera", action="store_true",
                        help="Camera + device preflight only, then exit")
    return parser


def load_contract(args) -> dict:
    if args.contract:
        with open(args.contract) as fh:
            return json.load(fh)
    try:
        return load_bundled_contract()
    except (OSError, ValueError):
        return {"contractVersion": SUPPORTED_CONTRACT}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    display = Display(args.host, settle_timeout_s=args.timeout_s)
    try:
        status = display.status()
    except CalibError as exc:
        print(f"error: cannot reach display at {args.host}: {exc}", file=sys.stderr)
        return 2
    print(f"display: {status.get('totalModules')} modules, "
          f"charset {status.get('charset')}, contract v{status.get('contractVersion')}")

    with Camera(args.camera_index) as camera:
        if args.check_camera:
            try:
                diag = camera.check_camera()
            except CameraError as exc:
                print(f"camera check FAILED: {exc}", file=sys.stderr)
                return 3
            print(f"camera OK: {diag}")
            print("device reachable, camera stable. Ready to calibrate.")
            return 0
        try:
            camera.check_camera()
        except CameraError as exc:
            print(f"camera check FAILED: {exc}", file=sys.stderr)
            return 3
        os.makedirs(args.photo_dir, exist_ok=True)
        calib = Calibrator(display, camera, photo_dir=args.photo_dir,
                           dwell_ms=args.dwell_ms, timeout_s=args.timeout_s,
                           max_phase=args.phase,
                           identity_thresh=args.identity_thresh,
                           relearn_templates=args.relearn_templates)
        report = calib.run()
    print(f"result: {report['result']} ({report.get('reason', '')})")
    print(f"photos + report.json in {args.photo_dir}")
    return 0 if report["result"] == "converged" else 1


if __name__ == "__main__":
    raise SystemExit(main())
