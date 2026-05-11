from collections import deque
from time import perf_counter

import cv2
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import iirpeak, lfilter


def _clip01(value):
    return float(max(0.0, min(1.0, value)))


def _bandpass_amps_time_domain(signal_1d, sample_rate_hz, centers_hz):
    """Measure band amplitudes with time-domain IIR bandpass filters.

    Bandwidths are derived from neighbor spacing and widened slightly so
    adjacent bands overlap for continuous coverage.
    """
    if signal_1d is None:
        return [0.0] * len(centers_hz)

    sig = np.asarray(signal_1d, dtype=np.float32)
    if sig.size < 8 or len(centers_hz) == 0:
        return [0.0] * len(centers_hz)

    sig = sig - float(np.mean(sig))
    fs = float(max(sample_rate_hz, 1e-6))
    nyq = 0.5 * fs

    centers = [float(c) for c in centers_hz]
    out = []
    n = len(centers)
    for i, fc in enumerate(centers):
        if fc <= 0.0 or fc >= (0.98 * nyq):
            out.append(0.0)
            continue

        if n == 1:
            spacing_l = max(0.5, fc * 0.5)
            spacing_r = spacing_l
        else:
            spacing_l = (fc - centers[i - 1]) if i > 0 else (centers[i + 1] - fc)
            spacing_r = (centers[i + 1] - fc) if i < (n - 1) else (fc - centers[i - 1])
            spacing_l = max(spacing_l, 1e-6)
            spacing_r = max(spacing_r, 1e-6)

        # Slightly >1 gives gentle overlap into neighboring bands.
        bw = 1.15 * min(spacing_l, spacing_r)
        bw = float(np.clip(bw, 0.15, 0.95 * nyq))

        q = float(max(fc / max(bw, 1e-6), 0.35))
        b, a = iirpeak(fc, q, fs=fs)
        y = lfilter(b, a, sig)

        # Drop early transient and use RMS amplitude of steady output.
        drop = min(max(0, int(0.2 * y.size)), max(0, y.size - 4))
        y_steady = y[drop:] if drop < y.size else y
        amp = float(np.sqrt(np.mean(np.square(y_steady)))) if y_steady.size else 0.0
        out.append(amp)

    return out


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
        self.last_static_contours = []
        self.prev_contour_boxes = []
        self.prev_contour_centroids = []
        self.last_slit_data = None
        self.last_flow = None
        self.last_flow_slow = None
        self.last_direction = 0.0
        self.last_flow_metrics = {
            "direction_deg": 0.0,
            "speed_norm": 0.0,
            "coherence": 0.0,
            "activity": 0.0,
            "fast_activity": 0.0,
            "fast_speed_norm": 0.0,
            "fast_direction_deg": 0.0,
            "fast_coherence": 0.0,
            "slow_activity": 0.0,
            "slow_speed_norm": 0.0,
            "slow_direction_deg": 0.0,
            "slow_coherence": 0.0,
        }
        # Multi-scale flow: slow scale compares frames flow_slow_interval apart.
        self.prev_flow_gray_slow = None
        self.flow_slow_interval = 8
        self._last_slow_metrics = None
        # Envelope follower for band amplitudes (fast attack, slow decay).
        self.band_env_attack_s = 0.1
        self.band_env_release_s = 2.0
        self.band_env_states = {}
        self.band_env_last_t = {}

        # Optional preprocessing filters inspired by water_slits3_correlate_q.py
        self.enable_temporal_change_filter = True
        self.screen_blend_mode = 1
        self.blur_mode = 1
        self.gain_mode = 0   # 0=off, 1=-25%, 2=+50%, 3=auto
        # Temporal change filter: first-order IIR lowpass with highpass output.
        # Computes: output = input - lowpass(input), emphasizing dynamic content.
        self.temporal_filter_seconds = 8.0  # IIR time constant (smoothing window)
        self.temporal_filter_output_mode = "change"  # "change" or "lowpass"
        self.prev_temporal_float = None     # IIR lowpass state
        self.prev_temporal_u8 = None        # uint8 view of lowpass state for subtraction
        self.temporal_change_u8 = None      # reusable output buffer for lowpass highpass
        self.enable_temporal_difference_filter = False
        self.prev_temporal_diff_frame = None
        self.temporal_diff_u8 = None
        self.temporal_diff_polarity = "positive"
        self.temporal_diff_auto_gain = False
        self.enable_threshold_filter = False
        self.enable_contour_motion_filter = True
        self.show_static_contours = True
        # Retained field name for compatibility; now interpreted as minimum straight-line length in pixels.
        self.contour_motion_threshold_px = 0.0
        x = np.arange(256, dtype=np.float32) / 255.0
        screen_vals = 1.0 - ((1.0 - x) * (1.0 - x))
        self.screen_blend_lut = np.clip(screen_vals * 255.0, 0.0, 255.0).astype(np.uint8).reshape((256, 1))
        screen_vals_2x = 1.0 - ((1.0 - screen_vals) * (1.0 - screen_vals))
        self.screen_blend_lut_2x = np.clip(screen_vals_2x * 255.0, 0.0, 255.0).astype(np.uint8).reshape((256, 1))
        self.preprocess_blur_kernel_small = (5, 5)
        self.preprocess_blur_kernel_large = (15, 15)

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
        self.spectral_update_hz = 4.0
        self.last_spectral_update_t = 0.0

        self.smoothed = {
            "wave_frequency_hz": 0.0,
            "freq_centroid_hz": 0.0,
            "bump_size_common": 0.0,
            "bump_size_spread": 0.0,
            "bump_size_max": 0.0,
            "bump_size_centroid": 0.0,
            "bump_shape_roundness": 0.0,
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
            self.prev_contour_boxes = []
            self.prev_contour_centroids = []
            self.last_static_contours = []

    def _apply_temporal_difference_filter(self, gray_img):
        if self.prev_temporal_diff_frame is None or self.prev_temporal_diff_frame.shape != gray_img.shape:
            self.prev_temporal_diff_frame = gray_img.copy()
            return np.zeros(gray_img.shape, dtype=np.uint8)

        if self.temporal_diff_u8 is None or self.temporal_diff_u8.shape != gray_img.shape:
            self.temporal_diff_u8 = np.empty_like(gray_img)

        if self.temporal_diff_polarity == "both":
            cv2.absdiff(gray_img, self.prev_temporal_diff_frame, dst=self.temporal_diff_u8)
        elif self.temporal_diff_polarity == "positive":
            cv2.subtract(gray_img, self.prev_temporal_diff_frame, dst=self.temporal_diff_u8)
        else:  # "negative"
            cv2.subtract(self.prev_temporal_diff_frame, gray_img, dst=self.temporal_diff_u8)

        self.prev_temporal_diff_frame = gray_img.copy()
        return self.temporal_diff_u8

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
            self.prev_flow_gray_slow = None
            self._last_slow_metrics = None
            self.last_flow = None
            self.last_flow_slow = None
            self.flow_frame_index = 0

    def set_mask_points(self, points):
        self.mask_points = points if points and len(points) == 4 else None

    @staticmethod
    def _contour_centroid_px(contour):
        moms = cv2.moments(contour)
        if moms['m00'] > 1e-6:
            return float(moms['m10'] / moms['m00']), float(moms['m01'] / moms['m00'])
        x, y, w, h = cv2.boundingRect(contour)
        return float(x + 0.5 * w), float(y + 0.5 * h)

    def _split_contours_by_linearity(self, contours):
        if not contours:
            return [], []

        min_line_len_px = float(self.contour_motion_threshold_px)
        if min_line_len_px <= 0.0:
            # U=0.0 explicitly disables line filtering.
            return list(contours), []

        blob_like = []
        straight_like = []

        for contour in contours:
            if contour is None or len(contour) < 2:
                blob_like.append(contour)
                continue

            pts = contour.reshape(-1, 2).astype(np.float32)
            if pts.shape[0] < 2:
                blob_like.append(contour)
                continue

            vx, vy, x0, y0 = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            direction = np.array([float(vx), float(vy)], dtype=np.float32)
            dir_norm = float(np.linalg.norm(direction))
            if dir_norm <= 1e-6:
                blob_like.append(contour)
                continue
            direction /= dir_norm
            normal = np.array([-direction[1], direction[0]], dtype=np.float32)

            origin = np.array([float(x0), float(y0)], dtype=np.float32)
            rel = pts - origin
            proj = rel @ direction
            perp = np.abs(rel @ normal)

            line_len = float(np.max(proj) - np.min(proj)) if proj.size else 0.0
            perp_p90 = float(np.percentile(perp, 90.0)) if perp.size else 1e9

            _, (w_rect, h_rect), _ = cv2.minAreaRect(contour)
            major = float(max(w_rect, h_rect, 1.0))
            minor = float(max(min(w_rect, h_rect), 1.0))
            aspect = major / minor

            # Straightness ratio is perpendicular deviation as a fraction of
            # the fitted line span (e.g. 0.10 means ~10% deviation).
            straightness_ratio = perp_p90 / max(line_len, 1.0)

            # Reject squiggles/curves that can still look elongated in minAreaRect.
            perimeter = float(cv2.arcLength(contour, True))
            perimeter_line_ratio = perimeter / max(2.0 * line_len, 1.0)

            # Require actual long straight edge support from polygonal contour.
            # This suppresses irregular blobs that span long distances in projection
            # but do not contain any genuinely long line segment.
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            longest_edge = 0.0
            if approx is not None and len(approx) >= 2:
                poly = approx.reshape(-1, 2).astype(np.float32)
                segs = np.roll(poly, -1, axis=0) - poly
                edge_lens = np.linalg.norm(segs, axis=1)
                if edge_lens.size:
                    longest_edge = float(np.max(edge_lens))

            # As U increases, also tighten geometric criteria to avoid keeping
            # similarly many false positives at larger minimum lengths.
            u_tighten = float(np.clip((min_line_len_px - 15.0) / 85.0, 0.0, 1.0))
            max_straightness_ratio = 0.12 - (0.05 * u_tighten)
            min_aspect_ratio = 2.3 + (1.2 * u_tighten)
            max_perimeter_ratio = 1.90 - (0.50 * u_tighten)
            max_width_to_length = 0.36 - (0.16 * u_tighten)
            min_long_edge = max(10.0, 0.45 * min_line_len_px)

            is_straight_line = (
                line_len >= min_line_len_px
                and straightness_ratio <= max_straightness_ratio
                and aspect >= min_aspect_ratio
                and perimeter_line_ratio <= max_perimeter_ratio
                and (minor / max(major, 1.0)) <= max_width_to_length
                and longest_edge >= min_long_edge
            )

            if is_straight_line:
                straight_like.append(contour)
            else:
                blob_like.append(contour)

        return blob_like, straight_like

    def _screen_blend_self(self, gray_img):
        # Screen blend via LUT avoids per-pixel float math each frame.
        if self.screen_blend_mode <= 0:
            return gray_img
        if self.screen_blend_mode == 1:
            return cv2.LUT(gray_img, self.screen_blend_lut)
        return cv2.LUT(gray_img, self.screen_blend_lut_2x)

    def _apply_temporal_change_filter(self, gray_img):
        # First-order IIR lowpass: y = alpha * x + (1 - alpha) * y_prev
        alpha = 1.0 / max(1.0, self.fps * self.temporal_filter_seconds)
        if self.prev_temporal_float is None or self.prev_temporal_float.shape != gray_img.shape:
            self.prev_temporal_float = gray_img.astype(np.float32)
            self.prev_temporal_u8 = gray_img.copy()
            self.temporal_change_u8 = np.zeros_like(gray_img)
            if self.temporal_filter_output_mode == "lowpass":
                return self.prev_temporal_u8
            # Return zeros on first frame in change mode (no change history yet)
            return self.temporal_change_u8
        if self.prev_temporal_u8 is None or self.prev_temporal_u8.shape != gray_img.shape:
            self.prev_temporal_u8 = np.empty_like(gray_img)
        if self.temporal_change_u8 is None or self.temporal_change_u8.shape != gray_img.shape:
            self.temporal_change_u8 = np.empty_like(gray_img)
        # OpenCV accepts uint8 input here; avoid a full-frame float conversion on every frame.
        cv2.accumulateWeighted(gray_img, self.prev_temporal_float, alpha)
        cv2.convertScaleAbs(self.prev_temporal_float, dst=self.prev_temporal_u8)
        if self.temporal_filter_output_mode == "lowpass":
            return self.prev_temporal_u8
        cv2.subtract(gray_img, self.prev_temporal_u8, dst=self.temporal_change_u8)
        return self.temporal_change_u8

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
            self.prev_temporal_float = None
            self.prev_temporal_u8 = None
            self.temporal_change_u8 = None

        if self.enable_temporal_difference_filter:
            t_temporal_diff = perf_counter()
            source = self._apply_temporal_difference_filter(source)
            stage_times["temporal_diff_ms"] = (perf_counter() - t_temporal_diff) * 1000.0
        else:
            self.prev_temporal_diff_frame = None
            self.prev_contour_boxes = []
            self.prev_contour_centroids = []
            self.last_static_contours = []
            self.temporal_diff_u8 = None

        if self.screen_blend_mode > 0:
            t_screen = perf_counter()
            source = self._screen_blend_self(source)
            stage_times["screen_blend_ms"] = (perf_counter() - t_screen) * 1000.0

        # Keep source tonal structure intact so temporal/screen filters are visually and analytically meaningful.
        roi_gray = cv2.bitwise_and(source, source, mask=mask)

        # Gain stage before blur and thresholding.
        if self.gain_mode == 0:
            gain_img = roi_gray
        elif self.gain_mode == 1:
            gain_img = cv2.convertScaleAbs(roi_gray, alpha=0.75, beta=0)
        elif self.gain_mode == 2:
            gain_img = cv2.convertScaleAbs(roi_gray, alpha=1.50, beta=0)
        else:  # autogain: stretch 95th percentile to ~220
            nz = roi_gray[roi_gray > 0]
            p95 = float(np.percentile(nz, 95)) if nz.size > 0 else 1.0
            auto_alpha = min(8.0, 220.0 / max(p95, 1.0))
            gain_img = cv2.convertScaleAbs(roi_gray, alpha=auto_alpha, beta=0)

        if self.blur_mode > 0:
            t_blur = perf_counter()
            kernel = self.preprocess_blur_kernel_small if self.blur_mode == 1 else self.preprocess_blur_kernel_large
            blur = cv2.GaussianBlur(gain_img, kernel, 0)
            stage_times["pre_blur_ms"] = (perf_counter() - t_blur) * 1000.0
        else:
            blur = gain_img

        preprocess_display = cv2.bitwise_and(blur, blur, mask=mask)

        if self.enable_threshold_filter:
            # Contour/threshold branch taps from blur stage as requested.
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            thresh = preprocess_display.copy()
        edges = cv2.Canny(blur, 40, 120)
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
        edges = cv2.bitwise_and(edges, edges, mask=mask)

        self.last_threshold = thresh
        self.last_edges = edges
        return mask, roi_gray, blur, thresh, edges, preprocess_display, stage_times

    def _analyze_contours(self, thresh, mask):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) >= self.contour_min_area]
        # Split into blob-like contours and long straight-line contours.
        blob_contours, straight_line_contours = self._split_contours_by_linearity(valid)
        # Keep existing key names for compatibility with current UI wiring.
        self.last_static_contours = straight_line_contours
        self.last_contours = blob_contours

        areas = np.array([cv2.contourArea(c) for c in blob_contours], dtype=np.float32)
        if areas.size == 0:
            return {
                "common": 0.0,
                "median": 0.0,
                "spread": 0.0,
                "max": 0.0,
                "centroid": 0.0,
                "shape_roundness": 0.0,
                "count": 0,
                "fill_ratio": 0.0,
                "stability": 0.0,
            }

        hist, bins = np.histogram(areas, bins=min(16, max(4, int(np.sqrt(areas.size) + 1))))
        mode_bin = int(np.argmax(hist))

        # Normalize contour area metrics to full image area for scale invariance.
        img_pixels = float(thresh.shape[0] * thresh.shape[1])
        area_norm = 1.0 / max(img_pixels, 1.0)

        common = float((bins[mode_bin] + bins[mode_bin + 1]) * 0.5) * area_norm
        median = float(np.median(areas)) * area_norm
        q75 = float(np.quantile(areas, 0.75)) * area_norm
        q25 = float(np.quantile(areas, 0.25)) * area_norm
        spread = max(0.0, q75 - q25)
        max_area = float(np.max(areas)) * area_norm
        # Size centroid: center of bump-size distribution (mean normalized contour area).
        centroid = float(np.mean(areas)) * area_norm

        # Shape roundness score: 1.0 for round contours, 0.0 for very line-like contours.
        roundness_weighted = 0.0
        roundness_weight_sum = 0.0
        for contour, area in zip(blob_contours, areas):
            if area <= 0.0:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            circularity = _clip01((4.0 * np.pi * float(area)) / (perimeter * perimeter))
            roundness_weighted += circularity * float(area)
            roundness_weight_sum += float(area)
        shape_roundness = (roundness_weighted / roundness_weight_sum) if roundness_weight_sum > 0.0 else 0.0

        mask_pixels = float(np.count_nonzero(mask))
        fill_ratio = float(np.sum(areas) / mask_pixels) if mask_pixels > 0 else 0.0

        cv = float(np.std(areas) / (np.mean(areas) + 1e-6))
        stability = _clip01(1.0 / (1.0 + cv))

        return {
            "common": common,
            "median": median,
            "spread": spread,
            "max": max_area,
            "centroid": centroid,
            "shape_roundness": shape_roundness,
            "count": int(areas.size),
            "fill_ratio": fill_ratio,
            "stability": stability,
        }

    def _slit_rows(self, h):
        if self.slit_count <= 1:
            return [h // 2]
        return [int((idx + 1) * h / (self.slit_count + 1)) for idx in range(self.slit_count)]

    @staticmethod
    def _gentle_edge_window(length, taper_fraction=0.2):
        if length <= 1:
            return np.ones(max(1, length), dtype=np.float32)

        taper = int(max(1, round(length * taper_fraction)))
        taper = min(taper, max(1, length // 2))
        win = np.ones(length, dtype=np.float32)

        if taper <= 1:
            return win

        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, taper, dtype=np.float32)))
        win[:taper] = ramp
        win[-taper:] = ramp[::-1]
        return win

    def _temporal_fft_from_history(self, history, sample_rate_hz=None):
        if len(history) < 12:
            return 0.0, np.array([]), np.array([])

        sig = np.array(history, dtype=np.float32)
        sig -= float(np.mean(sig))
        sig *= np.hanning(len(sig))

        ytf = np.abs(rfft(sig))
        fs = float(sample_rate_hz) if sample_rate_hz is not None else self.fps
        fs = max(1e-6, fs)
        xtf = rfftfreq(len(sig), 1.0 / fs)
        valid = xtf > 0.05
        peak = 0.0
        if np.any(valid):
            peak_idx = int(np.argmax(ytf[valid]))
            peak = float(xtf[valid][peak_idx])
        return peak, xtf, ytf

    @staticmethod
    def _quadrant_centers(h, w):
        row_top = h // 4
        row_bot = (3 * h) // 4
        col_left = w // 4
        col_right = (3 * w) // 4
        return [
            ("TL", row_top, col_left),
            ("TR", row_top, col_right),
            ("BL", row_bot, col_left),
            ("BR", row_bot, col_right),
        ]

    def _update_temporal_histories(self, roi_gray):
        h, w = roi_gray.shape
        centers = self._quadrant_centers(h, w)
        points = {}
        for i, (label, row, col) in enumerate(centers):
            patch = roi_gray[max(0, row - 2) : min(h, row + 3), max(0, col - 2) : min(w, col + 3)]
            self.slit_history[i].append(float(np.mean(patch)))
            points[label] = {"x": int(col), "y": int(row)}
        return points

    @staticmethod
    def _quadrant_spatial_summary(roi_gray):
        h, w = roi_gray.shape
        half_h = h // 2
        half_w = w // 2
        regions = {
            "UL": roi_gray[0:half_h, 0:half_w],
            "UR": roi_gray[0:half_h, half_w:w],
            "LL": roi_gray[half_h:h, 0:half_w],
            "LR": roi_gray[half_h:h, half_w:w],
        }

        out = {}
        for label, block in regions.items():
            if block.size == 0 or block.shape[1] < 8:
                out[label] = {"peak": 0.0, "band_low": 0.0, "band_mid": 0.0, "band_high": 0.0, "dominant": "-"}
                continue

            profile = np.mean(block.astype(np.float32), axis=0)
            profile -= float(np.mean(profile))
            profile *= np.hanning(len(profile))

            yf = np.abs(rfft(profile))
            xf = rfftfreq(len(profile), 1.0) * block.shape[1]

            if len(yf) > 1:
                peak_idx = int(np.argmax(yf[1:]) + 1)
                peak = float(xf[peak_idx])
            else:
                peak = 0.0

            low_mask = (xf >= 0.5) & (xf < 3.0)
            mid_mask = (xf >= 3.0) & (xf < 8.0)
            high_mask = (xf >= 8.0) & (xf <= 20.0)
            low_energy = float(np.sum(yf[low_mask]))
            mid_energy = float(np.sum(yf[mid_mask]))
            high_energy = float(np.sum(yf[high_mask]))
            total_energy = low_energy + mid_energy + high_energy + 1e-6

            band_vals = {
                "low": low_energy / total_energy,
                "mid": mid_energy / total_energy,
                "high": high_energy / total_energy,
            }
            dominant = max(band_vals, key=band_vals.get)

            out[label] = {
                "peak": peak,
                "band_low": band_vals["low"],
                "band_mid": band_vals["mid"],
                "band_high": band_vals["high"],
                "dominant": dominant,
            }

        return out

    def _analyze_slits(self, roi_gray, update_temporal_history=True):
        h, w = roi_gray.shape

        # Fixed slit positions at quadrant center lines (25% / 75%).
        row_top  = h // 4
        row_bot  = (3 * h) // 4
        col_left  = w // 4
        col_right = (3 * w) // 4
        slit_rows = [row_top, row_bot]
        slit_cols = [col_left, col_right]
        win_x = self._gentle_edge_window(w, taper_fraction=0.2)
        win_y = self._gentle_edge_window(h, taper_fraction=0.2)

        # --- Spatial: 2 horizontal slits (top / bottom) ---
        spatial_peaks = []
        low_energy = 0.0
        mid_energy = 0.0
        high_energy = 0.0
        spatial_xf = np.array([])
        spatial_yf_accum = None
        horizontal_spectra = []

        for row_idx, row in enumerate(slit_rows):
            slit_u8 = roi_gray[row, :].copy()
            slit = slit_u8.astype(np.float32)
            slit_demean = slit - float(np.mean(slit))
            slit_win = slit_demean * win_x
            yf = np.abs(rfft(slit_win))
            xf = rfftfreq(len(slit), 1.0) * w   # cycles per image width
            spatial_xf = xf
            spatial_bands_raw = _bandpass_amps_time_domain(slit_demean, float(w), [4.0, 10.0, 20.0])
            spatial_bands = self._apply_band_envelope(f"spatial_h_{row_idx}", spatial_bands_raw)
            horizontal_spectra.append({"row": int(row), "xf": xf, "yf": yf, "samples": slit_u8, "bands": spatial_bands})
            if spatial_yf_accum is None:
                spatial_yf_accum = np.zeros_like(yf, dtype=np.float32)
            spatial_yf_accum += yf.astype(np.float32)
            if len(yf) > 1:
                peak_idx = int(np.argmax(yf[1:]) + 1)
                spatial_peaks.append(float(xf[peak_idx]))
            low_mask  = (xf >= 0.5) & (xf < 3.0)
            mid_mask  = (xf >= 3.0) & (xf < 8.0)
            high_mask = (xf >= 8.0) & (xf <= 20.0)
            low_energy  += float(np.sum(yf[low_mask]))
            mid_energy  += float(np.sum(yf[mid_mask]))
            high_energy += float(np.sum(yf[high_mask]))

        # --- Spatial: 2 vertical slits (left / right) ---
        vertical_spectra = []
        for col_idx, col in enumerate(slit_cols):
            slit_u8 = roi_gray[:, col].copy()
            slit = slit_u8.astype(np.float32)
            slit_demean = slit - float(np.mean(slit))
            slit_win = slit_demean * win_y
            yf = np.abs(rfft(slit_win))
            xf_v = rfftfreq(len(slit), 1.0) * h  # cycles per image height
            spatial_bands_v_raw = _bandpass_amps_time_domain(slit_demean, float(h), [4.0, 10.0, 20.0])
            spatial_bands_v = self._apply_band_envelope(f"spatial_v_{col_idx}", spatial_bands_v_raw)
            vertical_spectra.append({"col": int(col), "xf": xf_v, "yf": yf, "samples": slit_u8, "bands": spatial_bands_v})
            if len(yf) > 1:
                peak_idx = int(np.argmax(yf[1:]) + 1)
                spatial_peaks.append(float(xf_v[peak_idx]))
            low_mask  = (xf_v >= 0.5) & (xf_v < 3.0)
            mid_mask  = (xf_v >= 3.0) & (xf_v < 8.0)
            high_mask = (xf_v >= 8.0) & (xf_v <= 20.0)
            low_energy  += float(np.sum(yf[low_mask]))
            mid_energy  += float(np.sum(yf[mid_mask]))
            high_energy += float(np.sum(yf[high_mask]))

        # --- Temporal: 4 quadrant center points (TL, TR, BL, BR) ---
        quadrant_centers = self._quadrant_centers(h, w)
        if update_temporal_history:
            quadrant_points = self._update_temporal_histories(roi_gray)
        else:
            quadrant_points = {label: {"x": int(col), "y": int(row)} for label, row, col in quadrant_centers}
        temporal_freqs = []
        temporal_xf = np.array([])
        temporal_yf = np.array([])
        quadrant_temporal = {}
        for i, (label, row, col) in enumerate(quadrant_centers):
            tpeak, xtf, ytf = self._temporal_fft_from_history(self.slit_history[i], sample_rate_hz=self.fps)
            temporal_bands_raw = _bandpass_amps_time_domain(self.slit_history[i], self.fps, [1.0, 2.5, 6.0])
            temporal_bands = self._apply_band_envelope(f"temporal_{label}", temporal_bands_raw)
            quadrant_temporal[label] = {"peak": tpeak, "xf": xtf, "yf": ytf, "bands": temporal_bands}
            if tpeak > 0.0:
                temporal_freqs.append(tpeak)
            if len(xtf) > 0:
                temporal_xf = xtf
                temporal_yf = ytf

        spatial_yf = (
            spatial_yf_accum / 2.0
            if spatial_yf_accum is not None
            else np.array([])
        )
        quadrant_summaries = self._quadrant_spatial_summary(roi_gray)
        total_energy = low_energy + mid_energy + high_energy + 1e-6

        temporal_centroid_hz = 0.0
        if temporal_xf.size > 0 and temporal_yf.size > 0:
            valid_t = temporal_xf > 0.05
            if np.any(valid_t):
                weights_t = temporal_yf[valid_t]
                weight_sum = float(np.sum(weights_t))
                if weight_sum > 1e-9:
                    temporal_centroid_hz = float(np.sum(temporal_xf[valid_t] * weights_t) / weight_sum)

        return {
            "spatial_peak_common": float(np.median(spatial_peaks)) if spatial_peaks else 0.0,
            "temporal_peak_common": float(np.median(temporal_freqs)) if temporal_freqs else 0.0,
            "temporal_centroid_hz": temporal_centroid_hz,
            "band_low":  low_energy  / total_energy,
            "band_mid":  mid_energy  / total_energy,
            "band_high": high_energy / total_energy,
            "spatial_xf": spatial_xf,
            "spatial_yf": spatial_yf,
            "quadrant_summaries": quadrant_summaries,
            "slit_rows": slit_rows,
            "slit_cols": slit_cols,
            "temporal_xf": temporal_xf,
            "temporal_yf": temporal_yf,
            "quadrant_temporal": quadrant_temporal,
            "quadrant_points": quadrant_points,
            "horizontal_spectra": horizontal_spectra,
            "vertical_spectra": vertical_spectra,
        }

    def _flow_metrics(self, flow, min_mag=0.25, speed_divisor=6.0, activity_divisor=8.0):
        """Extract direction/speed/coherence/activity scalars from a Farneback flow field."""
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        active = mag > min_mag
        if not np.any(active):
            return {"direction_deg": 0.0, "speed_norm": 0.0, "coherence": 0.0, "activity": 0.0}
        weights = mag[active]
        theta = ang[active]
        wx = float(np.sum(np.cos(theta) * weights))
        wy = float(np.sum(np.sin(theta) * weights))
        w_total = float(np.sum(weights)) + 1e-6
        direction = (np.degrees(np.arctan2(wy, wx)) + 360.0) % 360.0
        coherence = _clip01(float(np.sqrt(wx * wx + wy * wy) / w_total))
        speed_norm = _clip01(float(np.median(weights)) / speed_divisor)
        activity = _clip01(float(np.mean(weights)) / activity_divisor)
        return {
            "direction_deg": float(direction),
            "speed_norm": speed_norm,
            "coherence": coherence,
            "activity": activity,
        }

    def _analyze_flow(self, roi_gray):
        """Multi-scale optical flow: fast scale (adjacent frames) + slow scale (N frames apart)."""
        flow_data = dict(self.last_flow_metrics)

        # --- Fast scale: adjacent frames at flow_downscale resolution ---
        fast_w = max(32, int(roi_gray.shape[1] * self.flow_downscale))
        fast_h = max(24, int(roi_gray.shape[0] * self.flow_downscale))
        small_fast = cv2.resize(roi_gray, (fast_w, fast_h), interpolation=cv2.INTER_AREA)

        self.flow_frame_index += 1
        should_update = (self.flow_frame_index % self.flow_update_interval) == 0

        if self.prev_flow_gray is None:
            self.prev_flow_gray = small_fast
            self.prev_flow_gray_slow = None
            return flow_data

        if self.prev_flow_gray.shape != small_fast.shape:
            self.prev_flow_gray = small_fast
            self.prev_flow_gray_slow = None
            self._last_slow_metrics = None
            self.last_flow = None
            return flow_data

        if not should_update:
            self.prev_flow_gray = small_fast
            return flow_data

        fast_flow = cv2.calcOpticalFlowFarneback(
            self.prev_flow_gray,
            small_fast,
            None,
            self.flow_pyr_scale,
            self.flow_levels,
            self.flow_winsize,
            self.flow_iterations,
            self.flow_poly_n,
            self.flow_poly_sigma,
            self.flow_flags,
        )
        self.prev_flow_gray = small_fast
        self.last_flow = fast_flow
        fast_m = self._flow_metrics(fast_flow, min_mag=0.25, speed_divisor=6.0, activity_divisor=8.0)

        # --- Slow scale: compare frames flow_slow_interval apart at ~0.15x resolution ---
        # Captures large/slow movements that barely move between adjacent frames.
        slow_scale = max(0.05, self.flow_downscale * 0.3)
        slow_w = max(16, int(roi_gray.shape[1] * slow_scale))
        slow_h = max(12, int(roi_gray.shape[0] * slow_scale))
        small_slow = cv2.resize(roi_gray, (slow_w, slow_h), interpolation=cv2.INTER_AREA)

        is_slow_update = (self.flow_frame_index % self.flow_slow_interval) == 0
        if (
            is_slow_update
            and self.prev_flow_gray_slow is not None
            and self.prev_flow_gray_slow.shape == small_slow.shape
        ):
            slow_flow = cv2.calcOpticalFlowFarneback(
                self.prev_flow_gray_slow,
                small_slow,
                None,
                self.flow_pyr_scale,
                self.flow_levels,
                min(self.flow_winsize + 6, 41),  # wider window for large displacements
                self.flow_iterations,
                self.flow_poly_n,
                self.flow_poly_sigma,
                self.flow_flags,
            )
            self.last_flow_slow = slow_flow
            # Normalize to per-frame units by dividing divisors by flow_slow_interval.
            n = float(self.flow_slow_interval)
            self._last_slow_metrics = self._flow_metrics(
                slow_flow,
                min_mag=0.25 * n,
                speed_divisor=6.0 * n,
                activity_divisor=8.0 * n,
            )

        if is_slow_update or self.prev_flow_gray_slow is None:
            self.prev_flow_gray_slow = small_slow

        slow_m = self._last_slow_metrics

        # --- Combine fast and slow metrics ---
        if slow_m is not None and slow_m["activity"] > 0.0:
            fa = fast_m["activity"]
            sa = slow_m["activity"]
            total = fa + sa + 1e-6
            # Circular weighted mean for direction.
            fd = np.radians(fast_m["direction_deg"])
            sd = np.radians(slow_m["direction_deg"])
            dx = fa * np.cos(fd) + sa * np.cos(sd)
            dy = fa * np.sin(fd) + sa * np.sin(sd)
            direction = float((np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0)
            speed_norm = _clip01((fa * fast_m["speed_norm"] + sa * slow_m["speed_norm"]) / total)
            coherence = _clip01((fa * fast_m["coherence"] + sa * slow_m["coherence"]) / total)
            activity = _clip01(max(fa, sa))
        else:
            direction = fast_m["direction_deg"]
            speed_norm = fast_m["speed_norm"]
            coherence = fast_m["coherence"]
            activity = fast_m["activity"]

        self.last_direction = direction
        flow_data.update({
            "direction_deg": direction,
            "speed_norm": speed_norm,
            "coherence": coherence,
            "activity": activity,
            "fast_activity": fast_m["activity"],
            "fast_speed_norm": fast_m["speed_norm"],
            "fast_direction_deg": fast_m["direction_deg"],
            "fast_coherence": fast_m["coherence"],
            "slow_activity": slow_m["activity"] if slow_m else 0.0,
            "slow_speed_norm": slow_m["speed_norm"] if slow_m else 0.0,
            "slow_direction_deg": slow_m["direction_deg"] if slow_m else 0.0,
            "slow_coherence": slow_m["coherence"] if slow_m else 0.0,
        })
        self.last_flow_metrics = dict(flow_data)
        return flow_data

    def _ema(self, key, target, attack=0.35, release=0.1):
        current = self.smoothed[key]
        coeff = attack if target >= current else release
        self.smoothed[key] = (1.0 - coeff) * current + coeff * target
        return self.smoothed[key]

    def _apply_band_envelope(self, key, band_values):
        now = perf_counter()
        last_t = self.band_env_last_t.get(key)
        if last_t is None:
            dt = 1.0 / max(self.fps, 1.0)
        else:
            dt = max(1e-4, now - last_t)
        self.band_env_last_t[key] = now

        out = []
        for idx, target in enumerate(band_values):
            state_key = (key, idx)
            tgt = float(max(0.0, target))
            current = float(self.band_env_states.get(state_key, tgt))
            tau = self.band_env_attack_s if tgt >= current else self.band_env_release_s
            coeff = 1.0 - float(np.exp(-dt / max(tau, 1e-6)))
            val = current + coeff * (tgt - current)
            self.band_env_states[state_key] = val
            out.append(float(val))

        return out

    def _fuse(self, contours, slits, flow):
        raw = {
            "wave_frequency_hz": slits["temporal_peak_common"],
            "freq_centroid_hz": slits.get("temporal_centroid_hz", 0.0),
            "bump_size_common": contours["common"],
            "bump_size_spread": contours["spread"],
            "bump_size_max": contours.get("max", 0.0),
            "bump_size_centroid": contours.get("centroid", 0.0),
            "bump_shape_roundness": contours.get("shape_roundness", 0.0),
            "movement_direction_deg": flow["direction_deg"],
            "movement_speed_norm": flow["speed_norm"],
            "activity": _clip01(0.45 * flow["activity"] + 0.55 * min(1.0, contours["fill_ratio"] * 4.0)),
            "confidence": _clip01(0.4 * contours["stability"] + 0.3 * flow["coherence"] + 0.3 * (1.0 - abs(slits["band_high"] - 0.33))),
        }

        smoothed = {
            "wave_frequency_hz": self._ema("wave_frequency_hz", raw["wave_frequency_hz"], attack=0.28, release=0.08),
            "freq_centroid_hz": self._ema("freq_centroid_hz", raw["freq_centroid_hz"], attack=0.24, release=0.08),
            "bump_size_common": self._ema("bump_size_common", raw["bump_size_common"], attack=0.24, release=0.06),
            "bump_size_spread": self._ema("bump_size_spread", raw["bump_size_spread"], attack=0.2, release=0.05),
            "bump_size_max": self._ema("bump_size_max", raw["bump_size_max"], attack=0.24, release=0.06),
            "bump_size_centroid": self._ema("bump_size_centroid", raw["bump_size_centroid"], attack=0.2, release=0.08),
            "bump_shape_roundness": self._ema("bump_shape_roundness", raw["bump_shape_roundness"], attack=0.24, release=0.1),
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

        analysis_source = thresh if self.enable_threshold_filter else blur

        self.frame_index += 1
        run_full = self.frame_index % self.frame_skip == 0

        t1 = perf_counter()
        contour_data = self._analyze_contours(thresh, mask) if run_full else {
            "common": self.smoothed["bump_size_common"],
            "median": self.smoothed["bump_size_common"],
            "spread": self.smoothed["bump_size_spread"],
            "max": self.smoothed["bump_size_max"],
            "centroid": self.smoothed["bump_size_centroid"],
            "shape_roundness": self.smoothed["bump_shape_roundness"],
            "count": len(self.last_contours),
            "fill_ratio": 0.0,
            "stability": self.smoothed["confidence"],
        }
        timers["contours_ms"] = (perf_counter() - t1) * 1000.0

        t2 = perf_counter()
        # Tap spectral analysis from the threshold stage so thresholding affects
        # both slit spectra and temporal band measurements.
        if run_full:
            # Keep all temporal samples, independent of FFT recompute cadence.
            self._update_temporal_histories(analysis_source)
        spectral_interval_s = 1.0 / max(0.1, float(self.spectral_update_hz))
        now_s = perf_counter()
        run_spectral = (
            run_full and (
                self.last_slit_data is None
                or (now_s - self.last_spectral_update_t) >= spectral_interval_s
            )
        )
        if run_spectral:
            slit_data = self._analyze_slits(analysis_source, update_temporal_history=False)
            self.last_slit_data = slit_data
            self.last_spectral_update_t = now_s
        elif self.last_slit_data is not None:
            slit_data = self.last_slit_data
        else:
            slit_data = {
                "spatial_peak_common": 0.0,
                "temporal_peak_common": self.smoothed["wave_frequency_hz"],
                "temporal_centroid_hz": self.smoothed["freq_centroid_hz"],
                "band_low": 0.0,
                "band_mid": 0.0,
                "band_high": 0.0,
                "spatial_xf": np.array([]),
                "spatial_yf": np.array([]),
                "quadrant_summaries": {
                    "UL": {"peak": 0.0, "band_low": 0.0, "band_mid": 0.0, "band_high": 0.0, "dominant": "-"},
                    "UR": {"peak": 0.0, "band_low": 0.0, "band_mid": 0.0, "band_high": 0.0, "dominant": "-"},
                    "LL": {"peak": 0.0, "band_low": 0.0, "band_mid": 0.0, "band_high": 0.0, "dominant": "-"},
                    "LR": {"peak": 0.0, "band_low": 0.0, "band_mid": 0.0, "band_high": 0.0, "dominant": "-"},
                },
                "slit_rows": [],
                "slit_cols": [],
                "temporal_xf": np.array([]),
                "temporal_yf": np.array([]),
                "quadrant_temporal": {},
            }
        timers["slits_ms"] = (perf_counter() - t2) * 1000.0

        t3 = perf_counter()
        # Optical flow tracks gradients and local texture, so it is more stable
        # on the pre-threshold blurred signal than on a binary image.
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
            "analysis_source": analysis_source,
            "contours": self.last_contours,
            "static_contours": self.last_static_contours,
            "contour_data": contour_data,
            "slit_data": slit_data,
            "flow_data": flow_data,
            "raw": raw,
            "smoothed": smoothed,
            "timings": timers,
        }
