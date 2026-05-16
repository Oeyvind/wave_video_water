#!/usr/bin/env python3
"""Flow stability test: fast vs slow, circular vs axial std.

Axial std treats 0° and 180° as the same direction (back-and-forth assumption):
  double all angles → compute circular std → halve result.
This is more appropriate for oscillatory flow (waves going back and forth).
"""
import cv2
import numpy as np
from pathlib import Path
from wave_analysis import WaveAnalyzer

VIDEO_DIR = Path(r"C:\Projects\efx_experiments\wave_video_files")
TEST_FRAMES = 120
VIDEOS_TO_TEST = [
    "inderoy_pool_2.mp4",
    "inderoy_pool_3.mp4",
    "inderoy_pool_4.mp4",
    "inderoy_pool_5.mp4",
    "inderoy_pool_6.mp4",
]


def circular_std(angles_deg):
    """Standard circular std — treats 0° and 180° as opposite directions."""
    a = np.radians(np.asarray(angles_deg, dtype=np.float64))
    r = np.sqrt(np.sum(np.sin(a))**2 + np.sum(np.cos(a))**2) / max(len(a), 1)
    r = min(r, 1.0 - 1e-9)
    return float(np.degrees(np.sqrt(-2.0 * np.log(r))))


def axial_std(angles_deg):
    """Axial circular std — treats 0° and 180° as the SAME axis (back-and-forth).
    Method: double angles, compute circular std, halve result."""
    a2 = np.radians(np.asarray(angles_deg, dtype=np.float64) * 2.0)
    r = np.sqrt(np.sum(np.sin(a2))**2 + np.sum(np.cos(a2))**2) / max(len(a2), 1)
    r = min(r, 1.0 - 1e-9)
    return float(np.degrees(np.sqrt(-2.0 * np.log(r))) / 2.0)


print("Flow direction stability: circular vs axial std (back-and-forth assumption)\n")
hdr = f"{'Video':<25} {'Fast circ':>10} {'Fast axial':>11} {'Slow circ':>10} {'Slow axial':>11}  {'Best (axial)':<14}"
print(hdr)
print("=" * len(hdr))

for video_name in VIDEOS_TO_TEST:
    video_path = VIDEO_DIR / video_name
    if not video_path.exists():
        print(f"{video_name:<25} VIDEO NOT FOUND")
        continue

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"{video_name:<25} CANNOT OPEN")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        analyzer = WaveAnalyzer(fps=fps)

        fast_dirs, slow_dirs = [], []

        for _ in range(TEST_FRAMES):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = analyzer.analyze(gray)
            fd = result.get("flow_data", {})
            if fd and float(fd.get("fast_activity", 0.0)) > 0.05:
                fast_dirs.append(float(fd.get("fast_direction_deg", 0.0)))
                slow_dirs.append(float(fd.get("slow_direction_deg", 0.0)))

        cap.release()

        if len(fast_dirs) > 5:
            fc = circular_std(fast_dirs)
            fa = axial_std(fast_dirs)
            sc = circular_std(slow_dirs)
            sa = axial_std(slow_dirs)
            winner = "FAST" if fa <= sa else "SLOW"
            improvement = (min(fc, sc) - min(fa, sa))
            axial_gain = f"{winner} ({improvement:+.1f}°)"
            print(f"{video_name:<25} {fc:>9.1f}° {fa:>10.1f}° {sc:>9.1f}° {sa:>10.1f}°  {axial_gain}")
        else:
            print(f"{video_name:<25} INSUFFICIENT DATA ({len(fast_dirs)} active frames)")

    except Exception as e:
        print(f"{video_name:<25} ERROR: {str(e)[:50]}")

print("=" * len(hdr))
print("\nAxial std assumes 0° ≡ 180° (back-and-forth on same axis).")
print("Lower std = more stable / more reliable direction signal.")
