import sys
from pathlib import Path
import numpy as np
import cv2

repo = Path('c:/Cabbage_VST/CabbageEfx/wave_video_water')
sys.path.insert(0, str(repo))

from wave_analysis import WaveAnalyzer
from video_capture import get_frame

video_dir = Path('c:/Projects/efx_experiments/wave_video_files')
candidates = ['wave_sono_1.mp4', 'wave_sono_2.mp4', 'wave_sono2.mp4']

resolved = []
for name in candidates:
    p = video_dir / name
    if p.exists() and p not in resolved:
        resolved.append(p)

print('Resolved videos:')
for p in resolved:
    print(' ', p.name)

if not resolved:
    raise SystemExit('No target videos found')

def run_video(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    analyzer = WaveAnalyzer(fps=fps)
    analyzer.enable_gabor_analysis = False
    analyzer.set_quality(downscale=1.0, slit_count=6, contour_min_area=20.0, frame_skip=1)
    analyzer.enable_temporal_change_filter = True
    analyzer.temporal_filter_seconds = 0.5
    analyzer.temporal_filter_output_mode = 'change'
    analyzer.set_wave_shape_params(min_area_px=500.0, max_area_px=18000.0, min_confidence=0.45)

    frame_idx = 0
    counts = []
    tracks = {}

    while True:
        got = get_frame(cap, loop=False, target_size=(1280, 720))
        if got is None:
            break
        _frame, gray = got
        analysis = analyzer.analyze(gray)
        ws = analysis.get('wave_shapes_data') or {}
        dets = ws.get('detections', [])
        counts.append(len(dets))
        for d in dets:
            did = int(d.get('id', -1))
            cx, cy = d.get('center', (0.0, 0.0))
            conf = float(d.get('confidence', 0.0))
            rec = tracks.setdefault(did, {'samples': 0, 'sum_x': 0.0, 'sum_y': 0.0, 'max_conf': 0.0})
            rec['samples'] += 1
            rec['sum_x'] += float(cx)
            rec['sum_y'] += float(cy)
            rec['max_conf'] = max(rec['max_conf'], conf)
        frame_idx += 1

    cap.release()
    counts_arr = np.asarray(counts, dtype=np.float64) if counts else np.asarray([], dtype=np.float64)
    summary_tracks = []
    for tid, rec in tracks.items():
        if rec['samples'] < 3:
            continue
        summary_tracks.append((tid, rec['samples'], rec['sum_x']/rec['samples'], rec['sum_y']/rec['samples'], rec['max_conf']))
    summary_tracks.sort(key=lambda x: (x[1], x[4]), reverse=True)

    return {
        'frames': int(frame_idx),
        'avg_count': float(np.mean(counts_arr)) if counts_arr.size else 0.0,
        'max_count': int(np.max(counts_arr)) if counts_arr.size else 0,
        'tracks': summary_tracks[:10],
    }

for p in resolved:
    r = run_video(p)
    print(f"\n{p.name}")
    print(f"  frames={r['frames']} avg_detected={r['avg_count']:.3f} max_detected={r['max_count']}")
    if not r['tracks']:
        print('  tracks: none (with >=3 samples)')
    else:
        print('  tracks (id, samples, avg_center_x, avg_center_y, max_conf):')
        for t in r['tracks']:
            print(f"    {t[0]:>2d}, {t[1]:>4d}, {t[2]:>7.1f}, {t[3]:>7.1f}, {t[4]:.3f}")
