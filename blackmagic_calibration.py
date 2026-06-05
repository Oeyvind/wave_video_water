from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROPE_DIR = SCRIPT_DIR.parent / "midiplugs" / "domen_ai" / "Rope"


def _add_rope_path() -> None:
    if ROPE_DIR.exists() and str(ROPE_DIR) not in sys.path:
        sys.path.insert(0, str(ROPE_DIR))


_add_rope_path()

DEFAULT_IP_CANDIDATES = [
    "172.31.57.153",
    "172.31.75.153",
    "172.31.57.154",
    "172.31.75.154",
]
DEFAULT_CAMERA_CANDIDATES = list(range(1, 9))

try:
    from atem_auto_calibrate import run_extended_calibration
except Exception as exc:  # pragma: no cover
    run_extended_calibration = None  # type: ignore[assignment]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone Blackmagic / ATEM camera calibration")
    parser.add_argument("--ip", help="ATEM IP address")
    parser.add_argument("--camera", type=int, help="ATEM camera/input destination")
    parser.add_argument("--video-device", type=int, default=1, help="OpenCV video capture device index")
    parser.add_argument("--mode", choices=["static", "motion"], default="static", help="Calibration mode")
    parser.add_argument("--apply-best", action=argparse.BooleanOptionalAction, default=True, help="Apply the best found setting")
    parser.add_argument("--fallback-simple", action=argparse.BooleanOptionalAction, default=True, help="Fallback to simple calibration if extended fails")
    parser.add_argument("--collect-seconds", type=float, default=2.0, help="Initial state collection time")
    parser.add_argument("--settle-seconds", type=float, default=0.8, help="Wait after each camera change before sampling")
    parser.add_argument("--sample-seconds", type=float, default=1.6, help="Sampling duration per candidate")
    parser.add_argument("--write-timeout", type=float, default=2.5, help="Timeout for camera packet verification")
    parser.add_argument("--blur-size", type=int, default=5, help="Blur kernel size used in visibility scoring")
    parser.add_argument("--binary-thresh", type=int, default=15, help="Binary threshold used in visibility scoring")
    parser.add_argument("--capture-width", type=int, default=960, help="Requested camera capture width")
    parser.add_argument("--capture-height", type=int, default=540, help="Requested camera capture height")
    parser.add_argument("--gain-values", default="100,200,300,400,500,600", help="Comma-separated ISO values to test")
    args = parser.parse_args()

    if run_extended_calibration is None:
        print(json.dumps({"connected": False, "error": f"Missing dependency: {IMPORT_ERROR}"}, indent=2))
        return 2

    def choose_ip() -> str:
        env_ip = os.environ.get("ATEM_IP", "").strip()
        if args.ip:
            return args.ip.strip()
        if env_ip:
            return env_ip
        print("ATEM IP was not provided.")
        print("Known candidates:")
        for idx, candidate in enumerate(DEFAULT_IP_CANDIDATES, start=1):
            print(f"  {idx}. {candidate}")
        print(f"  {len(DEFAULT_IP_CANDIDATES) + 1}. Enter a custom IP")
        while True:
            choice = input("Select ATEM IP: ").strip()
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(DEFAULT_IP_CANDIDATES):
                    return DEFAULT_IP_CANDIDATES[choice_num - 1]
                if choice_num == len(DEFAULT_IP_CANDIDATES) + 1:
                    custom_ip = input("Enter ATEM IP: ").strip()
                    if custom_ip:
                        return custom_ip
            except ValueError:
                if choice:
                    return choice
            print("Please choose one of the listed options or type a valid IP address.")

    def choose_camera() -> int:
        env_camera = os.environ.get("ATEM_CAMERA", "").strip()
        if args.camera is not None:
            return int(args.camera)
        if env_camera:
            return int(env_camera)
        print("ATEM camera/input was not provided.")
        print("Selectable camera inputs:")
        for idx, candidate in enumerate(DEFAULT_CAMERA_CANDIDATES, start=1):
            print(f"  {idx}. Camera {candidate}")
        print(f"  {len(DEFAULT_CAMERA_CANDIDATES) + 1}. Enter a custom camera/input number")
        while True:
            choice = input("Select ATEM camera/input: ").strip()
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(DEFAULT_CAMERA_CANDIDATES):
                    return DEFAULT_CAMERA_CANDIDATES[choice_num - 1]
                if choice_num == len(DEFAULT_CAMERA_CANDIDATES) + 1:
                    custom_camera = input("Enter ATEM camera/input number: ").strip()
                    if custom_camera:
                        return int(custom_camera)
            except ValueError:
                if choice:
                    return int(choice)
            print("Please choose one of the listed options or type a valid camera/input number.")

    atem_ip = choose_ip()
    atem_camera = choose_camera()

    print(
        json.dumps(
            {
                "requested_capture": {
                    "width": args.capture_width,
                    "height": args.capture_height,
                },
                "atem_ip": atem_ip,
                "camera_input": atem_camera,
                "note": "This script applies camera settings in ATEM/Blackmagic control and leaves them on the device after exit.",
            },
            indent=2,
        )
    )

    output, status = run_extended_calibration(
        atem_ip=atem_ip,
        camera=atem_camera,
        video_device=args.video_device,
        target_width=args.capture_width,
        target_height=args.capture_height,
        collect_seconds=args.collect_seconds,
        gain_values=[int(value) for value in args.gain_values.split(",") if value.strip()],
        settle_seconds=args.settle_seconds,
        sample_seconds=args.sample_seconds,
        write_timeout=args.write_timeout,
        blur_size=args.blur_size,
        binary_thresh=args.binary_thresh,
        mode=args.mode,
        strategy="extended",
        fallback_simple=args.fallback_simple,
        apply_best=args.apply_best,
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())