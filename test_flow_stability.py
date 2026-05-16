#!/usr/bin/env python3
"""Test optical flow direction stability: fast vs slow scale across all test videos."""
import cv2
import numpy as np
from pathlib import Path
from wave_analysis import WaveAnalyzer

VIDEO_DIR = Path(r"C:\Projects\efx_experiments\wave_video_files")
TEST_FRAMES = 150  # Analyze first 150 frames per video


def circular_mean(angles_deg):
    """Compute mean direction from circular angles."""
    angles = np.radians(angles_deg)
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    mean = np.degrees(np.arctan2(sin_sum, cos_sum))
    return (mean + 360.0) % 360.0


def circular_std(angles_deg):
    """Compute circular standard deviation."""
    angles = np.radians(angles_deg)
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    r = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)
    if r >= 1.0:
        return 0.0
    return float(np.degrees(np.sqrt(-2.0 * np.log(r))))


def test_video(video_path):
    """Analyze a single video and return flow stability stats."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        analyzer = WaveAnalyzer(fps=fps)
        
        fast_dirs = []
        slow_dirs = []
        fast_cohs = []
        slow_cohs = []
        fast_acts = []
        slow_acts = []

        frame_count = 0
        while frame_count < TEST_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = analyzer.analyze(gray)

            flow_data = result.get("flow", {})
            if flow_data:
                fast_dirs.append(float(flow_data.get("fast_direction_deg", 0.0)))
                slow_dirs.append(float(flow_data.get("slow_direction_deg", 0.0)))
                fast_cohs.append(float(flow_data.get("fast_coherence", 0.0)))
                slow_cohs.append(float(flow_data.get("slow_coherence", 0.0)))
                fast_acts.append(float(flow_data.get("fast_activity", 0.0)))
                slow_acts.append(float(flow_data.get("slow_activity", 0.0)))

            frame_count += 1

        cap.release()

        if len(fast_dirs) < 10:
            return None

        # Filter out zero-activity frames for direction analysis
        fast_mask = [a > 0.1 for a in fast_acts]
        slow_mask = [a > 0.1 for a in slow_acts]
        
        fast_dirs_active = [d for d, m in zip(fast_dirs, fast_mask) if m]
        slow_dirs_active = [d for d, m in zip(slow_dirs, slow_mask) if m]

        stats = {
            "video": video_path.name,
            "frames_analyzed": len(fast_dirs),
            "fast_dir_mean": circular_mean(fast_dirs_active) if fast_dirs_active else 0.0,
            "fast_dir_std": circular_std(fast_dirs_active) if len(fast_dirs_active) > 1 else 0.0,
            "fast_coh_mean": float(np.mean(fast_cohs)) if fast_cohs else 0.0,
            "fast_act_mean": float(np.mean(fast_acts)) if fast_acts else 0.0,
            "slow_dir_mean": circular_mean(slow_dirs_active) if slow_dirs_active else 0.0,
            "slow_dir_std": circular_std(slow_dirs_active) if len(slow_dirs_active) > 1 else 0.0,
            "slow_coh_mean": float(np.mean(slow_cohs)) if slow_cohs else 0.0,
            "slow_act_mean": float(np.mean(slow_acts)) if slow_acts else 0.0,
            "fast_active_ratio": float(len(fast_dirs_active) / len(fast_dirs)) if fast_dirs else 0.0,
            "slow_active_ratio": float(len(slow_dirs_active) / len(slow_dirs)) if slow_dirs else 0.0,
        }
        return stats

    except Exception as e:
        print(f"Error processing {video_path.name}: {e}")
        return None


def main():
    print("Testing optical flow stability across all videos...\n")
    
    videos = sorted([f for f in VIDEO_DIR.glob("*.mp4")])
    results = []

    for video_path in videos:
        print(f"Testing {video_path.name}...", end=" ", flush=True)
        stats = test_video(video_path)
        if stats:
            results.append(stats)
            print(f"OK ({stats['frames_analyzed']} frames)")
        else:
            print("SKIPPED")

    print("\n" + "="*120)
    print(f"{'Video':<35} {'Fast Dir Std':<15} {'Slow Dir Std':<15} {'Fast Coh':<12} {'Slow Coh':<12} {'Fast Act':<12} {'Slow Act':<12}")
    print("="*120)

    for s in results:
        print(f"{s['video']:<35} {s['fast_dir_std']:>13.1f}° {s['slow_dir_std']:>13.1f}° "
              f"{s['fast_coh_mean']:>10.3f} {s['slow_coh_mean']:>10.3f} "
              f"{s['fast_act_mean']:>10.3f} {s['slow_act_mean']:>10.3f}")

    print("\n" + "="*120)
    print("SUMMARY:")
    print("="*120)
    
    if results:
        fast_stds = [s["fast_dir_std"] for s in results if s["fast_dir_std"] > 0]
        slow_stds = [s["slow_dir_std"] for s in results if s["slow_dir_std"] > 0]
        fast_cohs = [s["fast_coh_mean"] for s in results]
        slow_cohs = [s["slow_coh_mean"] for s in results]
        
        print(f"Fast direction stability (mean std dev): {np.mean(fast_stds):.1f}°")
        print(f"Slow direction stability (mean std dev): {np.mean(slow_stds):.1f}°")
        print(f"Fast coherence (mean): {np.mean(fast_cohs):.3f}")
        print(f"Slow coherence (mean): {np.mean(slow_cohs):.3f}")
        print(f"\nDifference (slow - fast dir std): {np.mean(slow_stds) - np.mean(fast_stds):+.1f}°")
        print(f"Difference (fast - slow coherence): {np.mean(fast_cohs) - np.mean(slow_cohs):+.3f}")
        
        if np.mean(slow_stds) > np.mean(fast_stds) * 1.5:
            print("\n⚠️  SLOW FLOW IS SIGNIFICANTLY NOISIER than fast flow.")
            print("   Recommendation: Consider disabling slow flow or using fast flow only.")
        elif np.mean(slow_cohs) < np.mean(fast_cohs) * 0.7:
            print("\n⚠️  SLOW FLOW COHERENCE IS MUCH LOWER than fast flow.")
            print("   Recommendation: Slow flow may be too noisy for reliable direction sensing.")
        else:
            print("\n✓ Both fast and slow flow appear reasonably reliable.")


if __name__ == "__main__":
    main()
