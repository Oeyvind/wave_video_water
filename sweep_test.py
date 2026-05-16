#!/usr/bin/env python3
"""
Parametric sweep test for Brattøra_1.mp4
Tests different flow rates, resolutions, and analysis methods
to find which configuration best detects weak diagonal large-scale waves.
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
import sys
from wave_analysis import WaveAnalyzer


@dataclass
class SweepConfig:
    """Configuration for a single test run."""
    name: str
    flow_update_interval: int
    flow_slow_interval: int
    downscale: float
    enable_lbp: bool = True
    enable_gabor: bool = True


@dataclass
class FrameMetrics:
    """Metrics extracted from a single frame."""
    frame_idx: int
    timestamp: float
    pyramid_centroid: float  # scale from 0-1
    lbp_roughness: float    # entropy
    gabor_wavelength_px: float
    flow_direction_deg: float
    flow_coherence: float
    flow_activity: float


class SweepRunner:
    def __init__(self, video_path: str, fps: float | None = None):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        src_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps is None:
            self.fps = src_fps if src_fps > 1.0 else 30.0
        else:
            self.fps = float(fps)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {self.video_path.name}")
        print(f"  Frames: {self.total_frames} @ {self.fps:.1f} fps ({self.total_frames/self.fps:.1f}s)")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        
        # Sample every N frames (aim for ~2-3 sec intervals for 30fps video)
        self.sample_interval = max(1, int(self.fps * 2.5))  # ~2.5 sec per sample
        print(f"  Sample interval: {self.sample_interval} frames (~{self.sample_interval/self.fps:.2f}s)")
    
    def extract_metrics(self, analyzer: WaveAnalyzer, frame_gray: np.ndarray, 
                       frame_idx: int) -> FrameMetrics:
        """Extract analysis metrics from a frame."""
        
        result = analyzer.analyze(frame_gray)
        
        # Pyramid centroid (0-1 scale from coarse to fine)
        pyramid_data = result.get("pyramid_data", {})
        pyramid_centroid = pyramid_data.get("scale_centroid", 0.5)
        
        # LBP roughness (entropy)
        lbp_data = result.get("lbp_data", {})
        lbp_roughness = lbp_data.get("roughness", 0.0)
        
        # Gabor wavelength (pixels)
        gabor_data = result.get("gabor_data", {})
        gabor_wavelength = gabor_data.get("dominant_wavelength_px", 10.0)
        
        # Flow metrics
        flow_data = result.get("flow_data", {})
        flow_direction = flow_data.get("direction_deg", 0.0)
        flow_coherence = flow_data.get("coherence", 0.0)
        flow_activity = flow_data.get("activity", 0.0)
        
        return FrameMetrics(
            frame_idx=frame_idx,
            timestamp=frame_idx / self.fps,
            pyramid_centroid=float(pyramid_centroid),
            lbp_roughness=float(lbp_roughness),
            gabor_wavelength_px=float(gabor_wavelength),
            flow_direction_deg=float(flow_direction),
            flow_coherence=float(flow_coherence),
            flow_activity=float(flow_activity)
        )
    
    def run_config(self, config: SweepConfig, verbose=True) -> List[FrameMetrics]:
        """Run analysis with a single configuration."""
        
        # Create analyzer
        analyzer = WaveAnalyzer(fps=self.fps)
        analyzer.flow_update_interval = config.flow_update_interval
        analyzer.flow_slow_interval = config.flow_slow_interval
        analyzer.downscale = config.downscale
        analyzer.flow_downscale = config.downscale
        analyzer.enable_lbp_analysis = config.enable_lbp
        analyzer.enable_gabor_analysis = config.enable_gabor
        
        if verbose:
            print(f"\n  Config: {config.name}")
            print(f"    Flow update: {config.flow_update_interval}, "
                  f"slow interval: {config.flow_slow_interval}")
            print(f"    Downscale: {config.downscale:.2f}, "
                  f"LBP: {config.enable_lbp}, Gabor: {config.enable_gabor}")
        
        metrics = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
        
        frame_idx = 0
        sample_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Always analyze each frame to keep temporal state valid.
            try:
                m = self.extract_metrics(analyzer, gray, frame_idx)

                # Sample at intervals for reporting/storage.
                if frame_idx % self.sample_interval == 0:
                    metrics.append(m)
                    sample_count += 1
                    if verbose and sample_count % 5 == 0:
                        print(f"    Frame {frame_idx}: pyr_c={m.pyramid_centroid:.3f}, "
                              f"lbp={m.lbp_roughness:.2f}, gabor={m.gabor_wavelength_px:.1f}px, "
                              f"flow_dir={m.flow_direction_deg:.0f}°, coh={m.flow_coherence:.2f}")
            except Exception as e:
                print(f"    Error at frame {frame_idx}: {e}")
            
            frame_idx += 1
        
        if verbose:
            print(f"    Collected {len(metrics)} samples")
        
        return metrics
    
    def compute_stats(self, metrics: List[FrameMetrics]) -> Dict:
        """Compute summary statistics from metrics."""
        
        if not metrics:
            return {}
        
        # Extract time series
        pyramid_centroid = np.array([m.pyramid_centroid for m in metrics])
        lbp_roughness = np.array([m.lbp_roughness for m in metrics])
        gabor_wavelength = np.array([m.gabor_wavelength_px for m in metrics])
        flow_direction = np.array([m.flow_direction_deg for m in metrics])
        flow_coherence = np.array([m.flow_coherence for m in metrics])
        flow_activity = np.array([m.flow_activity for m in metrics])
        
        # Compute temporal derivatives (finite differences)
        pyr_centroid_deriv = np.diff(pyramid_centroid)
        lbp_roughness_deriv = np.diff(lbp_roughness)
        gabor_wavelength_deriv = np.diff(gabor_wavelength)
        
        # Expected wave motion: upper-left to lower-right
        # That's roughly 225° (or -135°), so we check:
        # 1. How often flow direction is near 225° ± 45°
        # 2. How strong/coherent that direction is
        
        # Normalize directions to [0, 360)
        flow_dir_norm = flow_direction % 360.0
        
        # Check how many frames have flow pointing near 225° ± 45° (180° to 270°)
        wave_direction_range = ((flow_dir_norm >= 180) & (flow_dir_norm <= 270))
        wave_direction_rate = np.mean(wave_direction_range)
        
        # Compute mean direction for frames where it's in expected range
        if np.any(wave_direction_range):
            mean_wave_direction = np.mean(flow_dir_norm[wave_direction_range])
        else:
            mean_wave_direction = 0.0
        
        stats = {
            # Pyramid
            "pyr_centroid_mean": float(np.mean(pyramid_centroid)),
            "pyr_centroid_std": float(np.std(pyramid_centroid)),
            "pyr_centroid_deriv_mean": float(np.mean(np.abs(pyr_centroid_deriv))),
            "pyr_centroid_deriv_std": float(np.std(np.abs(pyr_centroid_deriv))),
            "pyr_centroid_deriv_max": float(np.max(np.abs(pyr_centroid_deriv))) if len(pyr_centroid_deriv) > 0 else 0.0,
            
            # LBP
            "lbp_roughness_mean": float(np.mean(lbp_roughness)),
            "lbp_roughness_std": float(np.std(lbp_roughness)),
            "lbp_roughness_deriv_mean": float(np.mean(np.abs(lbp_roughness_deriv))),
            "lbp_roughness_deriv_std": float(np.std(np.abs(lbp_roughness_deriv))),
            "lbp_roughness_deriv_max": float(np.max(np.abs(lbp_roughness_deriv))) if len(lbp_roughness_deriv) > 0 else 0.0,
            
            # Gabor
            "gabor_wavelength_mean": float(np.mean(gabor_wavelength)),
            "gabor_wavelength_std": float(np.std(gabor_wavelength)),
            "gabor_wavelength_deriv_mean": float(np.mean(np.abs(gabor_wavelength_deriv))),
            "gabor_wavelength_deriv_std": float(np.std(np.abs(gabor_wavelength_deriv))),
            "gabor_wavelength_deriv_max": float(np.max(np.abs(gabor_wavelength_deriv))) if len(gabor_wavelength_deriv) > 0 else 0.0,
            
            # Flow
            "flow_direction_mean": float(np.mean(flow_direction)),
            "flow_direction_std": float(np.std(flow_direction)),
            "flow_coherence_mean": float(np.mean(flow_coherence)),
            "flow_coherence_std": float(np.std(flow_coherence)),
            "flow_activity_mean": float(np.mean(flow_activity)),
            "flow_activity_std": float(np.std(flow_activity)),
            
            # Wave detection metrics
            "wave_direction_rate": float(wave_direction_rate),  # % of frames with direction near 225°
            "mean_wave_direction": float(mean_wave_direction),
            "flow_coherence_when_wave_direction": float(np.mean(flow_coherence[wave_direction_range])) if np.any(wave_direction_range) else 0.0,
        }
        
        return stats
    
    def run_sweep(self) -> Dict[str, Dict]:
        """Run the full parametric sweep."""
        
        # Define test configurations
        configs = [
            # Baseline
            SweepConfig("baseline", flow_update_interval=1, flow_slow_interval=8, downscale=0.5),
            
            # Vary flow update interval (faster response)
            SweepConfig("fast_update_1", flow_update_interval=1, flow_slow_interval=8, downscale=0.5),
            SweepConfig("fast_update_2", flow_update_interval=2, flow_slow_interval=8, downscale=0.5),
            SweepConfig("fast_update_4", flow_update_interval=4, flow_slow_interval=8, downscale=0.5),
            
            # Vary slow flow interval (longer period for large waves)
            SweepConfig("slow_interval_6", flow_update_interval=1, flow_slow_interval=6, downscale=0.5),
            SweepConfig("slow_interval_12", flow_update_interval=1, flow_slow_interval=12, downscale=0.5),
            SweepConfig("slow_interval_18", flow_update_interval=1, flow_slow_interval=18, downscale=0.5),
            SweepConfig("slow_interval_24", flow_update_interval=1, flow_slow_interval=24, downscale=0.5),
            
            # Vary downscale (resolution)
            SweepConfig("downscale_100", flow_update_interval=1, flow_slow_interval=8, downscale=1.0),
            SweepConfig("downscale_075", flow_update_interval=1, flow_slow_interval=8, downscale=0.75),
            SweepConfig("downscale_050", flow_update_interval=1, flow_slow_interval=8, downscale=0.5),
            
            # Combined: coarse + slow
            SweepConfig("coarse_slow", flow_update_interval=1, flow_slow_interval=18, downscale=0.75),
            
            # Combined: fine + fast
            SweepConfig("fine_fast", flow_update_interval=2, flow_slow_interval=12, downscale=1.0),
        ]
        
        results = {}
        
        for config in configs:
            metrics = self.run_config(config, verbose=True)
            stats = self.compute_stats(metrics)
            results[config.name] = {
                "config": vars(config),
                "stats": stats,
                "metrics": [vars(m) for m in metrics],
            }
        
        return results
    
    def print_summary(self, results: Dict[str, Dict]):
        """Print summary of all configurations."""
        
        print("\n" + "="*100)
        print("SWEEP SUMMARY - Best Detectors for Diagonal Large-Scale Waves (Upper-Left → Lower-Right)")
        print("="*100)
        
        # Rank by different metrics
        metrics_to_rank = [
            ("pyramid_centroid_deriv_max", "Pyramid Centroid Change (max)", "higher"),
            ("lbp_roughness_deriv_max", "LBP Roughness Change (max)", "higher"),
            ("gabor_wavelength_deriv_max", "Gabor Wavelength Change (max)", "higher"),
            ("flow_coherence_mean", "Flow Coherence (mean)", "higher"),
            ("wave_direction_rate", "Frames with ~225° Flow Direction", "higher"),
            ("flow_coherence_when_wave_direction", "Coherence when flow ~225°", "higher"),
            ("flow_activity_mean", "Flow Activity (mean)", "higher"),
        ]
        
        for metric_key, metric_name, ranking in metrics_to_rank:
            print(f"\n{metric_name}:")
            print("-" * 80)
            
            # Extract metric for each config
            config_scores = []
            for config_name, result in results.items():
                score = result["stats"].get(metric_key, 0.0)
                config_scores.append((config_name, score))
            
            # Sort
            if ranking == "higher":
                config_scores.sort(key=lambda x: x[1], reverse=True)
            else:
                config_scores.sort(key=lambda x: x[1])
            
            # Print top 5
            for i, (name, score) in enumerate(config_scores[:5], 1):
                print(f"  {i}. {name:25s} {score:8.4f}")
        
        print("\n" + "="*100)
        print("Detailed Stats by Configuration:")
        print("="*100)
        
        for config_name in sorted(results.keys()):
            result = results[config_name]
            cfg = result["config"]
            stats = result["stats"]
            
            print(f"\n{config_name}:")
            print(f"  Flow: update_interval={cfg['flow_update_interval']}, "
                  f"slow_interval={cfg['flow_slow_interval']}, "
                  f"downscale={cfg['downscale']}")
            print(f"  Pyramid Centroid:")
            print(f"    Mean: {stats.get('pyr_centroid_mean', 0):.3f}, "
                  f"Std: {stats.get('pyr_centroid_std', 0):.3f}, "
                  f"Max change: {stats.get('pyr_centroid_deriv_max', 0):.4f}")
            print(f"  LBP Roughness:")
            print(f"    Mean: {stats.get('lbp_roughness_mean', 0):.2f}, "
                  f"Std: {stats.get('lbp_roughness_std', 0):.2f}, "
                  f"Max change: {stats.get('lbp_roughness_deriv_max', 0):.2f}")
            print(f"  Gabor Wavelength:")
            print(f"    Mean: {stats.get('gabor_wavelength_mean', 0):.1f}px, "
                  f"Std: {stats.get('gabor_wavelength_std', 0):.1f}px, "
                  f"Max change: {stats.get('gabor_wavelength_deriv_max', 0):.1f}px")
            print(f"  Flow:")
            print(f"    Direction: {stats.get('flow_direction_mean', 0):.0f}° "
                  f"(±{stats.get('flow_direction_std', 0):.0f}°)")
            print(f"    Coherence: {stats.get('flow_coherence_mean', 0):.3f}")
            print(f"    Activity: {stats.get('flow_activity_mean', 0):.3f}")
            print(f"  Wave Detection:")
            print(f"    Direction ~225°: {stats.get('wave_direction_rate', 0)*100:.1f}% of frames")
            print(f"    Mean direction (when detected): {stats.get('mean_wave_direction', 0):.0f}°")
            print(f"    Coherence when wave detected: {stats.get('flow_coherence_when_wave_direction', 0):.3f}")


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "Brattøra_1.mp4"

    # Check if video exists
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return

    runner = SweepRunner(video_path)
    results = runner.run_sweep()

    runner.print_summary(results)

    # Save detailed results to JSON
    output_path = Path(f"sweep_results_{Path(video_path).stem}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
