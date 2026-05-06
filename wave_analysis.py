from collections import deque
from time import perf_counter

import cv2
import numpy as np
from scipy.fft import rfft, rfftfreq


def _clip01(value):
    return float(max(0.0, min(1.0, value)))


class WaveAnalyzer:
    """Realtime wave analyzer with contour, spectral, and optical-flow engines."""

    def __init__(self, fps=30.0):
        self.fps = max(1.0, float(fps))
        self.downscale = 0.5
        self.slit_count = 4
        self.contour_min_area = 30.0
        self.frame_skip = 1
        self.show_mask = True
        self.prev_flow_gray = None
        self.frame_index = 0
        self.flow_frame_index = 0

        self.mask_points = None
        self.last_threshold = None
        self.last_edges = None
        self.last_contours = []
        self.last_slit_data = None
        self.last_flow = None
        self.last_direction = 0.0
        self.last_flow_metrics = {
            "direction_deg": 0.0,
            "speed_norm": 0.0,
            "coherence": 0.0,
            "activity": 0.0,
        }

        # Optional preprocessing filters inspired by water_slits3_correlate_q.py
        self.enable_temporal_change_filter = True
        self.enable_screen_blend_equalization = True
        self.enable_preprocess_blur = True
        # Temporal bandpass: output = fast_EMA - slow_EMA.
        # slow_EMA tracks the static background (DC / still reflections).
        # fast_EMA pre-smooths per-frame noise while still following wave motion.
        # The difference isolates the wave frequency band between the two cutoffs.
        self.temporal_filter_seconds = 8.0        # slow EMA: ~0.02 Hz cutoff
        self.temporal_filter_fast_seconds = 0.25  # fast EMA: ~0.6 Hz noise floor
        self.temporal_blur_kernel = (6, 6)
        self.prev_temporal_lpf_float = None       # slow EMA state
        self.prev_temporal_fast_float = None      # fast EMA state
        self.enable_temporal_difference_filter = True
        self.prev_temporal_diff_frame = None
        self.temporal_diff_polarity = "both"
        self.temporal_diff_auto_gain = True
        self.temporal_diff_target_level = 180.0
        self.temporal_diff_max_gain = 8.0

        # Flow quality controls (independent from main analysis quality).
        self.flow_downscale = 0.5
        self.flow_update_interval = 1
        self.flow_pyr_scale = 0.5
        self.flow_levels = 3
        self.flow_winsize = 15
        self.flow_iterations = 3
        self.flow_poly_n = 5
        self.flow_poly_sigma = 1.2
        self.flow_flags = 0

        history_len = int(self.fps * 4.0)
        self.slit_history = [deque(maxlen=history_len) for _ in range(self.slit_count)]

        self.smoothed = {
            "wave_frequency_hz": 0.0,
            "bump_size_common": 0.0,
            "bump_size_spread": 0.0,
            "movement_direction_deg": 0.0,
            "movement_speed_norm": 0.0,
            "activity": 0.0,
            "confidence": 0.0,
        }

    def set_quality(self, downscale=None, slit_count=None, contour_min_area=None, frame_skip=None):
        quality_changed = False
        if downscale is not None:
            new_downscale = float(max(0.2, min(1.0, downscale)))
            if abs(new_downscale - self.downscale) > 1e-9:
                self.downscale = new_downscale
                quality_changed = True
        if slit_count is not None:
            new_count = int(max(3, min(6, slit_count)))
            if new_count != self.slit_count:
                history_len = int(self.fps * 4.0)
                old_vals = [list(buf) for buf in self.slit_history]
                self.slit_count = new_count
                self.slit_history = [deque(maxlen=history_len) for _ in range(self.slit_count)]
                for idx in range(min(len(old_vals), self.slit_count)):
                    self.slit_history[idx].extend(old_vals[idx][-history_len:])
        if contour_min_area is not None:
            self.contour_min_area = float(max(1.0, contour_min_area))
        if frame_skip is not None:
            self.frame_skip = int(max(1, frame_skip))

        if quality_changed:
            # Force optical-flow warm restart if frame scale changes.
            self.prev_flow_gray = None
            self.last_flow = None
            self.prev_temporal_diff_frame = None

    def _apply_temporal_difference_filter(self, gray_img):
        current = gray_img.astype(np.int16)
        if self.prev_temporal_diff_frame is None or self.prev_temporal_diff_frame.shape != current.shape:
            self.prev_temporal_diff_frame = current.copy()
            return np.zeros(gray_img.shape, dtype=np.uint8)

        delta = current - self.prev_temporal_diff_frame
        self.prev_temporal_diff_frame = current.copy()

        if self.temporal_diff_polarity == "positive":
            out = np.clip(delta, 0, 255)
        elif self.temporal_diff_polarity == "negative":
            out = np.clip(-delta, 0, 255)
        else:
            out = np.clip(np.abs(delta), 0, 255)

        out_u8 = out.astype(np.uint8)
        if not self.temporal_diff_auto_gain:
            return out_u8

        nonzero = out_u8[out_u8 > 0]
        if nonzero.size == 0:
            return out_u8

        p95 = float(np.percentile(nonzero, 95.0))
        if p95 < 1e-6:
            return out_u8

        gain = min(self.temporal_diff_max_gain, max(1.0, self.temporal_diff_target_level / p95))
        boosted = np.clip(out_u8.astype(np.float32) * gain, 0.0, 255.0)
        return boosted.astype(np.uint8)

    def set_flow_quality(
        self,
        flow_downscale=None,
        flow_update_interval=None,
        flow_pyr_scale=None,
        flow_levels=None,
        flow_winsize=None,
        flow_iterations=None,
        flow_poly_n=None,
        flow_poly_sigma=None,
        flow_flags=None,
    ):
        changed = False

        if flow_downscale is not None:
            new_val = float(max(0.05, min(1.0, flow_downscale)))
            if abs(new_val - self.flow_downscale) > 1e-9:
                self.flow_downscale = new_val
                changed = True

        if flow_update_interval is not None:
            new_val = int(max(1, flow_update_interval))
            if new_val != self.flow_update_interval:
                self.flow_update_interval = new_val
                changed = True

        if flow_pyr_scale is not None:
            new_val = float(max(0.1, min(0.95, flow_pyr_scale)))
            if abs(new_val - self.flow_pyr_scale) > 1e-9:
                self.flow_pyr_scale = new_val
                changed = True

        if flow_levels is not None:
            new_val = int(max(1, min(8, flow_levels)))
            if new_val != self.flow_levels:
                self.flow_levels = new_val
                changed = True

        if flow_winsize is not None:
            new_val = int(max(5, min(41, flow_winsize)))
            if new_val != self.flow_winsize:
                self.flow_winsize = new_val
                changed = True

        if flow_iterations is not None:
            new_val = int(max(1, min(10, flow_iterations)))
            if new_val != self.flow_iterations:
                self.flow_iterations = new_val
                changed = True

        if flow_poly_n is not None:
            # OpenCV Farneback supports poly_n values typically 5 or 7.
            new_val = 5 if int(flow_poly_n) <= 5 else 7
            if new_val != self.flow_poly_n:
                self.flow_poly_n = new_val
                changed = True

        if flow_poly_sigma is not None:
            new_val = float(max(1.0, min(2.0, flow_poly_sigma)))
            if abs(new_val - self.flow_poly_sigma) > 1e-9:
                self.flow_poly_sigma = new_val
                changed = True

        if flow_flags is not None:
            new_val = int(max(0, flow_flags))
            if new_val != self.flow_flags:
                self.flow_flags = new_val
                changed = True

        if changed:
            self.prev_flow_gray = None
            self.last_flow = None
            self.flow_frame_index = 0

    def set_mask_points(self, points):
        self.mask_points = points if points and len(points) == 4 else None

    @staticmethod
    def _screen_blend_self(gray_img):
        base_float = gray_img.astype(np.float32) / 255.0
        result_float = 1.0 - ((1.0 - base_float) * (1.0 - base_float))
        return np.clip(result_float * 255.0, 0.0, 255.0).astype(np.uint8)

    def _apply_temporal_change_filter(self, gray_img):
        blurred = cv2.blur(gray_img, self.temporal_blur_kernel)
        current_float = blurred.astype(np.float32)

        # --- slow EMA: converges to static background over ~temporal_filter_seconds ---
        alpha_slow = 1.0 / max(1.0, self.fps * self.temporal_filter_seconds)
        if self.prev_temporal_lpf_float is None or self.prev_temporal_lpf_float.shape != current_float.shape:
            self.prev_temporal_lpf_float = current_float.copy()
        self.prev_temporal_lpf_float = (
            alpha_slow * current_float
            + (1.0 - alpha_slow) * self.prev_temporal_lpf_float
        )

        # --- fast EMA: smooths per-frame noise, still follows wave motion ---
        alpha_fast = 1.0 / max(1.0, self.fps * self.temporal_filter_fast_seconds)
        if self.prev_temporal_fast_float is None or self.prev_temporal_fast_float.shape != current_float.shape:
            self.prev_temporal_fast_float = current_float.copy()
        self.prev_temporal_fast_float = (
            alpha_fast * current_float
            + (1.0 - alpha_fast) * self.prev_temporal_fast_float
        )

        # Bandpass = fast_EMA - slow_EMA: retains only the wave-frequency band.
        diff = np.clip(self.prev_temporal_fast_float - self.prev_temporal_lpf_float, 0.0, 255.0).astype(np.uint8)
        return diff

    def _mask_and_preprocess(self, gray):
        stage_times = {
            "temporal_filter_ms": 0.0,
            "temporal_diff_ms": 0.0,
            "screen_blend_ms": 0.0,
            "pre_blur_ms": 0.0,
        }

        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        if self.mask_points is None:
            cv2.rectangle(mask, (0, 0), (w - 1, h - 1), 255, -1)
        else:
            pts = np.array(self.mask_points, dtype=np.int32)
            cv2.fillConvexPoly(mask, pts, 255)

        source = gray
        if self.enable_temporal_change_filter:
            t_temporal = perf_counter()
            source = self._apply_temporal_change_filter(source)
            stage_times["temporal_filter_ms"] = (perf_counter() - t_temporal) * 1000.0

        else:
            self.prev_temporal_lpf_float = None
            self.prev_temporal_fast_float = None

        if self.enable_temporal_difference_filter:
            t_temporal_diff = perf_counter()
            source = self._apply_temporal_difference_filter(source)
            stage_times["temporal_diff_ms"] = (perf_counter() - t_temporal_diff) * 1000.0
        else:
            self.prev_temporal_diff_frame = None

        if self.enable_screen_blend_equalization:
            t_screen = perf_counter()
            source = self._screen_blend_self(source)
            stage_times["screen_blend_ms"] = (perf_counter() - t_screen) * 1000.0

        # Keep source tonal structure intact so temporal/screen filters are visually and analytically meaningful.
        roi_gray = cv2.bitwise_and(source, source, mask=mask)
        if self.enable_preprocess_blur:
            t_blur = perf_counter()
            blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
            stage_times["pre_blur_ms"] = (perf_counter() - t_blur) * 1000.0
        else:
            blur = roi_gray

        preprocess_display = cv2.bitwise_and(blur, blur, mask=mask)

        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(blur, 40, 120)
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
        edges = cv2.bitwise_and(edges, edges, mask=mask)

        self.last_threshold = thresh
        self.last_edges = edges
        return mask, roi_gray, blur, thresh, edges, preprocess_display, stage_times

    def _analyze_contours(self, thresh, mask):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) >= self.contour_min_area]
        self.last_contours = valid

        areas = np.array([cv2.contourArea(c) for c in valid], dtype=np.float32)
        if areas.size == 0:
            return {
                "common": 0.0,
                "median": 0.0,
                "spread": 0.0,
                "count": 0,
                "fill_ratio": 0.0,
                "stability": 0.0,
            }

        hist, bins = np.histogram(areas, bins=min(16, max(4, int(np.sqrt(areas.size) + 1))))
        mode_bin = int(np.argmax(hist))
        common = float((bins[mode_bin] + bins[mode_bin + 1]) * 0.5)
        median = float(np.median(areas))
        q75 = float(np.quantile(areas, 0.75))
        q25 = float(np.quantile(areas, 0.25))
        spread = max(0.0, q75 - q25)

        mask_pixels = float(np.count_nonzero(mask))
        fill_ratio = float(np.sum(areas) / mask_pixels) if mask_pixels > 0 else 0.0

        cv = float(np.std(areas) / (np.mean(areas) + 1e-6))
        stability = _clip01(1.0 / (1.0 + cv))

        return {
            "common": common,
            "median": median,
            "spread": spread,
            "count": int(areas.size),
            "fill_ratio": fill_ratio,
            "stability": stability,
        }

    def _slit_rows(self, h):
        if self.slit_count <= 1:
            return [h // 2]
        return [int((idx + 1) * h / (self.slit_count + 1)) for idx in range(self.slit_count)]

    def _analyze_slits(self, roi_gray):
        h, w = roi_gray.shape
        rows = self._slit_rows(h)
        spatial_peaks = []
        low_energy = 0.0
        mid_energy = 0.0
        high_energy = 0.0

        for idx, row in enumerate(rows):
            slit = roi_gray[row : row + 1, :].flatten().astype(np.float32)
            slit -= float(np.mean(slit))
            slit *= np.hanning(len(slit))

            yf = np.abs(rfft(slit))
            xf = rfftfreq(len(slit), 1.0)

            if len(yf) > 1:
                peak_idx = int(np.argmax(yf[1:]) + 1)
                spatial_peaks.append(float(xf[peak_idx] * w))

            self.slit_history[idx].append(float(np.mean(roi_gray[max(0, row - 1) : min(h, row + 2), :])))

            low_mask = (xf * w >= 0.5) & (xf * w < 3.0)
            mid_mask = (xf * w >= 3.0) & (xf * w < 8.0)
            high_mask = (xf * w >= 8.0) & (xf * w <= 20.0)
            low_energy += float(np.sum(yf[low_mask]))
            mid_energy += float(np.sum(yf[mid_mask]))
            high_energy += float(np.sum(yf[high_mask]))

        temporal_freqs = []
        temporal_xf = np.array([])
        temporal_yf = np.array([])
        for history in self.slit_history:
            if len(history) < 12:
                continue
            sig = np.array(history, dtype=np.float32)
            sig -= float(np.mean(sig))
            sig *= np.hanning(len(sig))

            ytf = np.abs(rfft(sig))
            xtf = rfftfreq(len(sig), 1.0 / self.fps)
            valid = xtf > 0.05
            if np.any(valid):
                peak_idx = int(np.argmax(ytf[valid]))
                temporal_freqs.append(float(xtf[valid][peak_idx]))
            temporal_xf = xtf
            temporal_yf = ytf

        total_energy = low_energy + mid_energy + high_energy + 1e-6
        return {
            "spatial_peak_common": float(np.median(spatial_peaks)) if spatial_peaks else 0.0,
            "temporal_peak_common": float(np.median(temporal_freqs)) if temporal_freqs else 0.0,
            "band_low": low_energy / total_energy,
            "band_mid": mid_energy / total_energy,
            "band_high": high_energy / total_energy,
            "temporal_xf": temporal_xf,
            "temporal_yf": temporal_yf,
        }

    def _analyze_flow(self, roi_gray):
        flow_data = dict(self.last_flow_metrics)

        small = cv2.resize(
            roi_gray,
            (
                max(32, int(roi_gray.shape[1] * self.flow_downscale)),
                max(24, int(roi_gray.shape[0] * self.flow_downscale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

        self.flow_frame_index += 1
        should_update = (self.flow_frame_index % self.flow_update_interval) == 0

        if self.prev_flow_gray is None:
            self.prev_flow_gray = small
            return flow_data

        if self.prev_flow_gray.shape != small.shape:
            # Downscale/ROI change can alter dimensions between frames.
            self.prev_flow_gray = small
            self.last_flow = None
            return flow_data

        if not should_update:
            self.prev_flow_gray = small
            return flow_data

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_flow_gray,
            small,
            None,
            self.flow_pyr_scale,
            self.flow_levels,
            self.flow_winsize,
            self.flow_iterations,
            self.flow_poly_n,
            self.flow_poly_sigma,
            self.flow_flags,
        )
        self.prev_flow_gray = small
        self.last_flow = flow

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        active = mag > 0.25
        if not np.any(active):
            return flow_data

        weights = mag[active]
        theta = ang[active]
        x = float(np.sum(np.cos(theta) * weights))
        y = float(np.sum(np.sin(theta) * weights))

        direction = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
        coherence = np.sqrt(x * x + y * y) / (float(np.sum(weights)) + 1e-6)
        speed_norm = _clip01(float(np.median(weights)) / 6.0)
        activity = _clip01(float(np.mean(weights)) / 8.0)

        self.last_direction = float(direction)
        flow_data.update(
            {
                "direction_deg": float(direction),
                "speed_norm": speed_norm,
                "coherence": _clip01(float(coherence)),
                "activity": activity,
            }
        )
        self.last_flow_metrics = dict(flow_data)
        return flow_data

    def _ema(self, key, target, attack=0.35, release=0.1):
        current = self.smoothed[key]
        coeff = attack if target >= current else release
        self.smoothed[key] = (1.0 - coeff) * current + coeff * target
        return self.smoothed[key]

    def _fuse(self, contours, slits, flow):
        raw = {
            "wave_frequency_hz": slits["temporal_peak_common"],
            "bump_size_common": contours["common"],
            "bump_size_spread": contours["spread"],
            "movement_direction_deg": flow["direction_deg"],
            "movement_speed_norm": flow["speed_norm"],
            "activity": _clip01(0.45 * flow["activity"] + 0.55 * min(1.0, contours["fill_ratio"] * 4.0)),
            "confidence": _clip01(0.4 * contours["stability"] + 0.3 * flow["coherence"] + 0.3 * (1.0 - abs(slits["band_high"] - 0.33))),
        }

        smoothed = {
            "wave_frequency_hz": self._ema("wave_frequency_hz", raw["wave_frequency_hz"], attack=0.28, release=0.08),
            "bump_size_common": self._ema("bump_size_common", raw["bump_size_common"], attack=0.24, release=0.06),
            "bump_size_spread": self._ema("bump_size_spread", raw["bump_size_spread"], attack=0.2, release=0.05),
            "movement_direction_deg": raw["movement_direction_deg"],
            "movement_speed_norm": self._ema("movement_speed_norm", raw["movement_speed_norm"], attack=0.32, release=0.12),
            "activity": self._ema("activity", raw["activity"], attack=0.3, release=0.1),
            "confidence": self._ema("confidence", raw["confidence"], attack=0.2, release=0.08),
        }
        self.smoothed["movement_direction_deg"] = raw["movement_direction_deg"]
        return raw, smoothed

    def analyze(self, gray):
        timers = {}
        t0 = perf_counter()
        mask, roi_gray, blur, thresh, edges, preprocess_display, preprocess_stage_times = self._mask_and_preprocess(gray)
        timers["preprocess_ms"] = (perf_counter() - t0) * 1000.0
        timers.update(preprocess_stage_times)

        self.frame_index += 1
        run_full = self.frame_index % self.frame_skip == 0

        t1 = perf_counter()
        contour_data = self._analyze_contours(thresh, mask) if run_full else {
            "common": self.smoothed["bump_size_common"],
            "median": self.smoothed["bump_size_common"],
            "spread": self.smoothed["bump_size_spread"],
            "count": len(self.last_contours),
            "fill_ratio": 0.0,
            "stability": self.smoothed["confidence"],
        }
        timers["contours_ms"] = (perf_counter() - t1) * 1000.0

        t2 = perf_counter()
        # Tap spectral analysis after blur, as requested.
        if run_full:
            slit_data = self._analyze_slits(blur)
            self.last_slit_data = slit_data
        elif self.last_slit_data is not None:
            slit_data = self.last_slit_data
        else:
            slit_data = {
                "spatial_peak_common": 0.0,
                "temporal_peak_common": self.smoothed["wave_frequency_hz"],
                "band_low": 0.0,
                "band_mid": 0.0,
                "band_high": 0.0,
                "temporal_xf": np.array([]),
                "temporal_yf": np.array([]),
            }
        timers["slits_ms"] = (perf_counter() - t2) * 1000.0

        t3 = perf_counter()
        # Tap optical flow after blur as parallel branch from the same signal.
        flow_data = self._analyze_flow(blur)
        timers["flow_ms"] = (perf_counter() - t3) * 1000.0

        t4 = perf_counter()
        raw, smoothed = self._fuse(contour_data, slit_data, flow_data)
        timers["fusion_ms"] = (perf_counter() - t4) * 1000.0
        timers["total_ms"] = sum(timers.values())

        return {
            "mask": mask,
            "roi_gray": roi_gray,
            "threshold": thresh,
            "edges": edges,
            "preprocess_display": preprocess_display,
            "contours": self.last_contours,
            "contour_data": contour_data,
            "slit_data": slit_data,
            "flow_data": flow_data,
            "raw": raw,
            "smoothed": smoothed,
            "timings": timers,
        }