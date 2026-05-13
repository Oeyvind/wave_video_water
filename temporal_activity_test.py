import json
from pathlib import Path

import cv2
import numpy as np

from wave_analysis import WaveAnalyzer


def summarize(values):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def run_video(video_path, max_frames=500):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0

    analyzer = WaveAnalyzer(fps=fps)

    global_temporal = []
    upper_temporal = []
    lower_temporal = []
    quadrant_means = {"UL": [], "UR": [], "LL": [], "LR": []}

    frames = 0
    while frames < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = analyzer.analyze(gray)
        pd = result.get("pyramid_data", {})

        gt = float(pd.get("temporal_activity", 0.0))
        qt = pd.get("quadrant_temporal_bands", {})

        q_level = {}
        for q in ("UL", "UR", "LL", "LR"):
            qbands = np.asarray(qt.get(q, [0.0] * 5), dtype=np.float32)
            qmean = float(np.mean(qbands)) if qbands.size else 0.0
            q_level[q] = qmean
            quadrant_means[q].append(qmean)

        top = 0.5 * (q_level["UL"] + q_level["UR"])
        bottom = 0.5 * (q_level["LL"] + q_level["LR"])

        global_temporal.append(gt)
        upper_temporal.append(top)
        lower_temporal.append(bottom)

        frames += 1

    cap.release()

    g = np.asarray(global_temporal, dtype=np.float32)
    u = np.asarray(upper_temporal, dtype=np.float32)
    l = np.asarray(lower_temporal, dtype=np.float32)

    upper_gt_lower_ratio = float(np.mean(u > l)) if u.size and l.size else 0.0
    upper_over_lower_mean_ratio = float((np.mean(u) + 1e-9) / (np.mean(l) + 1e-9)) if u.size and l.size else 0.0

    return {
        "video": str(video_path),
        "fps": float(fps),
        "frames_analyzed": int(frames),
        "global_temporal": summarize(global_temporal),
        "upper_temporal": summarize(upper_temporal),
        "lower_temporal": summarize(lower_temporal),
        "upper_gt_lower_ratio": upper_gt_lower_ratio,
        "upper_over_lower_mean_ratio": upper_over_lower_mean_ratio,
        "quadrants": {q: summarize(vals) for q, vals in quadrant_means.items()},
    }


def main():
    source_dir = Path(r"C:/Projects/efx_experiments/wave_video_files")
    videos = [
        source_dir / "Brattøra_1.mp4",
        source_dir / "Nidelv_surface1.mp4",
    ]

    reports = []
    for vp in videos:
        if not vp.exists():
            raise FileNotFoundError(f"Video not found: {vp}")
        reports.append(run_video(vp, max_frames=600))

    print(json.dumps({"reports": reports}, indent=2))


if __name__ == "__main__":
    main()
