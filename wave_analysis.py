from collections import deque
from time import perf_counter

import cv2
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import iirpeak, lfilter

from wavelength_detector import detect_wavelength


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

    PYRAMID_BAND_COUNT = 3

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
        self.last_wavelength = None
        self.last_wavelength_metrics = {
            "wavelength_px": None,
            "confidence": 0.0,
            "horizontal_wavelength": None,
            "vertical_wavelength": None,
            "quadrants": {},
        }
        self.wl_use_median = True
        # Fixed-length median pre-filter + 2 Hz IIR lowpass for wavelength jitter removal.
        self._wl_history = deque(maxlen=5)
        self._wl_lp_state = None  # IIR lowpass state (None = uninitialised)
        self._wl_q_history = {
            "UL": deque(maxlen=5),
            "UR": deque(maxlen=5),
            "LL": deque(maxlen=5),
            "LR": deque(maxlen=5),
        }
        self._wl_q_lp_state = {
            "UL": None,
            "UR": None,
            "LL": None,
            "LR": None,
        }
        self.prev_pyramid_bands = None
        self.pyramid_temporal_activity = 0.0
        self.pyramid_temporal_band_activity = [0.0] * self.PYRAMID_BAND_COUNT
        self.prev_quadrant_bands = {}
        self.quadrant_temporal_band_activity = {}
        # Intermediate default: higher texture cap for better yellow-band fidelity
        # while retaining lower CPU than full-resolution analysis.
        self.pyramid_max_dim = 640
        self.pyramid_update_interval = 2
        self._last_pyramid_data = None
        # Centroid renormalization bounds (derived from clip-set percentiles).
        # Keep raw centroid for flow adaptation and expose renormalized variants
        # for display/OSC so the visible/control range spans 0..1 more fully.
        self.pyramid_centroid_clip = {
            "global_spatial": (0.2, 0.8),
            "global_temporal": (0.2, 0.8),
            "quadrant_spatial": (0.2, 0.8),
            "quadrant_temporal": (0.2, 0.8),
        }
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
            "fast_direction_quality": 0.0,
            "slow_direction_quality": 0.0,
            "direction_quality": 0.0,
            "direction_source_label": "fast",
            "adaptive_direction_deg": 0.0,
            "adaptive_speed_norm": 0.0,
            "adaptive_coherence": 0.0,
            "adaptive_activity": 0.0,
            "directional_source": "fast",
            "directional_target_deg": 225.0,
            "directional_global": {},
            "directional_quadrants": {},
            "directional_stripes": {},
            "directional_best_quadrant": "-",
            "directional_best_quadrant_support": 0.0,
            "directional_best_stripe": "-",
            "directional_best_stripe_support": 0.0,
        }
        # Multi-scale flow: slow scale compares frames flow_slow_interval apart.
        self.prev_flow_gray_slow = None
        self.flow_slow_interval = 4
        self._flow_interval_smooth = 4.0   # EMA-smoothed float target for adaptive interval
        self.enable_auto_flow_update_interval = True
        # Axial mode: fold direction to 0-180° (treats back-and-forth as same axis).
        self.flow_axial_mode = False
        self._flow_update_interval_smooth = 1.0
        self.flow_update_interval_min = 1
        self.flow_update_interval_max = 6
        self._flow_dir_history = {
            "fast": deque(maxlen=5),
            "slow": deque(maxlen=5),
            "blend": deque(maxlen=5),
            "adaptive": deque(maxlen=5),
        }
        self._flow_dir_unwrapped = {
            "fast": None,
            "slow": None,
            "blend": None,
            "adaptive": None,
        }
        self.flow_direction_target_deg = 225.0
        self.flow_direction_tolerance_deg = 45.0
        self.flow_direction_stripe_count = 5
        self.enable_flow_directional_scoring = True
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

        # Optional texture descriptor stages.
        self.enable_lbp_analysis = True
        self.enable_gabor_analysis = False
        # LBP triangle compound mapping parameters.
        self.lbp_order_center = 0.66
        self.lbp_order_width = 0.50
        self.lbp_chaos_entropy_exp = 0.60

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

    def set_lbp_compound_tuning(self, order_center=None, order_width=None, chaos_entropy_exp=None):
        """Tune LBP smooth/order/chaos corner mapping parameters."""
        if order_center is not None:
            self.lbp_order_center = float(np.clip(float(order_center), 0.0, 1.0))
        if order_width is not None:
            self.lbp_order_width = float(np.clip(float(order_width), 0.05, 1.0))
        if chaos_entropy_exp is not None:
            self.lbp_chaos_entropy_exp = float(np.clip(float(chaos_entropy_exp), 0.5, 4.0))

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
                self._flow_update_interval_smooth = float(new_val)
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
        # Always compute Otsu binary — captures broad bright blobs (e.g. foam crests)
        # that have gradual gradients and are invisible to Canny.
        _, thresh_otsu_c = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
        thresh_otsu_c = cv2.bitwise_and(thresh_otsu_c, thresh_otsu_c, mask=mask)
        edges = cv2.bitwise_and(edges, edges, mask=mask)
        edges_dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        # Combined binary: Canny edges trace fine wave texture; Otsu covers broad bright shapes.
        contour_binary = cv2.bitwise_or(edges_dilated, thresh_otsu_c)

        self.last_threshold = thresh
        self.last_edges = edges
        return mask, roi_gray, blur, thresh, edges, contour_binary, preprocess_display, stage_times

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

    @staticmethod
    def _pyramid_band_energies(gray_img, normalize=True):
        """Compute 3-band texture energies from a Gaussian/Laplacian pyramid.

        Band 0 is a broader band centered on the former yellow scale and
        blended with neighboring orange/green scales. Bands 1-2 preserve the
        former blue/purple coarse lowpass bands.
        """
        if gray_img is None or gray_img.size == 0:
            return [0.0, 0.0, 0.0]

        base = gray_img.astype(np.float32) / 255.0
        levels = [base]
        for _ in range(5):
            prev = levels[-1]
            if prev.shape[0] < 8 or prev.shape[1] < 8:
                break
            levels.append(cv2.pyrDown(prev))

        fine_laps = [0.0, 0.0, 0.0]
        lap_count = min(3, len(levels) - 1)
        for i in range(lap_count):
            up = cv2.pyrUp(levels[i + 1], dstsize=(levels[i].shape[1], levels[i].shape[0]))
            band = levels[i] - up
            fine_laps[i] = float(np.mean(np.abs(band)))

        # Broaden the first band around former yellow using neighboring scales.
        broad_weights = np.asarray([0.25, 0.5, 0.25], dtype=np.float32)
        fine_arr = np.asarray(fine_laps, dtype=np.float32)
        active = fine_arr > 0.0
        if np.any(active):
            w = broad_weights * active.astype(np.float32)
            broad_band = float(np.dot(fine_arr, w) / (float(np.sum(w)) + 1e-9))
        else:
            broad_band = 0.0

        # Preserve the former blue/purple coarse bands from deep lowpass levels.
        coarse_band = 0.0
        extra_coarse_band = 0.0
        if len(levels) >= 2:
            coarse_band = float(np.std(levels[-2]))
            extra_coarse_band = float(np.std(levels[-1]))
        elif len(levels) == 1:
            coarse_band = float(np.std(levels[0]))

        energies = [broad_band, coarse_band, extra_coarse_band]

        if not normalize:
            return [float(e) for e in energies]

        total = float(sum(energies)) + 1e-9
        return [float(e / total) for e in energies]

    @staticmethod
    def _resize_for_texture_analysis(gray_img, max_dim=320):
        if gray_img is None or gray_img.size == 0:
            return gray_img
        h, w = gray_img.shape[:2]
        largest = max(h, w)
        if largest <= max_dim:
            return gray_img
        scale = float(max_dim) / float(largest)
        new_w = max(16, int(round(w * scale)))
        new_h = max(16, int(round(h * scale)))
        return cv2.resize(gray_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _compute_lbp_codes(gray_img, max_dim=320):
        """Resize *gray_img* and compute the 8-bit LBP codes array (H×W uint8).

        This is the expensive O(pixels) step.  Returns ``None`` when the
        image is too small or empty.  Split out so callers that need both
        global and per-quadrant statistics can run this once and slice the
        resulting array, calling :meth:`_lbp_stats_from_codes` cheaply on
        each region.
        """
        if gray_img is None or gray_img.size == 0:
            return None
        view = WaveAnalyzer._resize_for_texture_analysis(gray_img, max_dim=max_dim)
        if min(view.shape[:2]) < 3:
            return None
        padded = np.pad(view.astype(np.uint8), 1, mode="edge")
        center = padded[1:-1, 1:-1]
        neighbors = [
            padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:],
            padded[1:-1, 2:], padded[2:, 2:], padded[2:, 1:-1],
            padded[2:, :-2], padded[1:-1, :-2],
        ]
        codes = np.zeros_like(center, dtype=np.uint8)
        for bit, neighbor in enumerate(neighbors):
            codes |= ((neighbor >= center).astype(np.uint8) << bit)
        return codes

    def _lbp_stats_from_codes(self, codes):
        """Compute all LBP statistics from a pre-computed codes array.

        Accepts any 2-D uint8 array — the full frame or a spatial slice for
        a quadrant.  This is cheap compared with code computation and can be
        called multiple times on sub-regions of the same codes array.
        """
        if codes is None or codes.size == 0:
            return {
                "lbp_mean": 0.0, "lbp_std": 0.0, "lbp_entropy": 0.0,
                "lbp_uniform_ratio": 0.0, "lbp_roughness": 0.0,
                "lbp_dominant_code": 0, "lbp_dominant_ratio": 0.0,
                "lbp_smooth": 0.0, "lbp_order": 0.0, "lbp_chaos": 0.0,
                "lbp_histogram_16": [0.0] * 16,
            }

        hist256 = np.bincount(codes.ravel(), minlength=256).astype(np.float32)
        hist_sum = float(np.sum(hist256)) + 1e-9
        hist256 /= hist_sum
        # Compact view: 16 grouped bins from 256 codes (16 codes per bin).
        hist16 = np.asarray([float(np.sum(chunk)) for chunk in np.array_split(hist256, 16)], dtype=np.float32)

        bits = ((codes[..., None] >> np.arange(8, dtype=np.uint8)) & 1).astype(np.uint8)
        transitions = np.sum(bits != np.roll(bits, -1, axis=2), axis=2)
        uniform_ratio_raw = float(np.mean(transitions <= 2))

        dominant_code = int(np.argmax(hist256))
        dominant_ratio = float(hist256[dominant_code])
        entropy_raw = float(-np.sum(hist256 * np.log2(hist256 + 1e-9)))
        lbp_mean = float(np.mean(codes)) / 255.0
        lbp_std = float(np.std(codes)) / 255.0
        roughness_raw = float(1.0 - dominant_ratio)

        # Stretch low-contrast ranges so values below ~0.3 become more expressive.
        # Applies to all three base measures before compound metrics are derived.
        def _stretch01(v):
            return float(np.clip((v - 0.3) / 0.7, 0.0, 1.0))

        roughness = _stretch01(roughness_raw)
        uniform_ratio = _stretch01(uniform_ratio_raw)
        entropy_norm = _stretch01(entropy_raw / 8.0)
        entropy = entropy_norm * 8.0

        # Triangle corner activations (smooth/order/chaos), then normalize.
        roughness_mid = float(max(0.0, 1.0 - (abs(roughness - self.lbp_order_center) / self.lbp_order_width)))
        chaos_entropy = float(np.clip(entropy_norm, 0.0, 1.0) ** self.lbp_chaos_entropy_exp)
        smooth_raw = float(uniform_ratio * (1.0 - roughness) * (1.0 - entropy_norm))
        order_raw = float(uniform_ratio * (1.0 - entropy_norm) * roughness_mid)
        chaos_raw = float(chaos_entropy * roughness * (1.0 - uniform_ratio))
        sum_raw = smooth_raw + order_raw + chaos_raw + 1e-9
        smooth = smooth_raw / sum_raw
        order = order_raw / sum_raw
        chaos = chaos_raw / sum_raw

        return {
            "lbp_mean": lbp_mean,
            "lbp_std": lbp_std,
            "lbp_entropy": entropy,
            "lbp_uniform_ratio": uniform_ratio,
            "lbp_roughness": roughness,
            "lbp_dominant_code": dominant_code,
            "lbp_dominant_ratio": dominant_ratio,
            "lbp_smooth": float(smooth),
            "lbp_order": float(order),
            "lbp_chaos": float(chaos),
            "lbp_histogram_16": [float(v) for v in hist16],
        }

    def _analyze_lbp_texture(self, gray_img):
        """Dense LBP summary for local roughness and micro-texture stability."""
        return self._lbp_stats_from_codes(self._compute_lbp_codes(gray_img))

    def _analyze_gabor_texture(self, gray_img):
        """Small Gabor bank for oriented wave/ripple energy."""
        if gray_img is None or gray_img.size == 0:
            return {
                "orientations_deg": [0.0, 45.0, 90.0, 135.0],
                "wavelengths_px": [4.0, 8.0, 16.0],
                "orientation_energy": [0.0, 0.0, 0.0, 0.0],
                "wavelength_energy": [0.0, 0.0, 0.0],
                "response_grid": [[0.0, 0.0, 0.0] for _ in range(4)],
                "dominant_orientation_deg": 0.0,
                "dominant_wavelength_px": 0.0,
                "gabor_anisotropy": 0.0,
                "gabor_energy": 0.0,
            }

        view = self._resize_for_texture_analysis(gray_img, max_dim=256)
        if min(view.shape[:2]) < 9:
            return {
                "orientations_deg": [0.0, 45.0, 90.0, 135.0],
                "wavelengths_px": [4.0, 8.0, 16.0],
                "orientation_energy": [0.0, 0.0, 0.0, 0.0],
                "wavelength_energy": [0.0, 0.0, 0.0],
                "response_grid": [[0.0, 0.0, 0.0] for _ in range(4)],
                "dominant_orientation_deg": 0.0,
                "dominant_wavelength_px": 0.0,
                "gabor_anisotropy": 0.0,
                "gabor_energy": 0.0,
            }

        work = cv2.GaussianBlur(view.astype(np.float32), (3, 3), 0)
        orientations = [0.0, 45.0, 90.0, 135.0]
        wavelengths = [4.0, 8.0, 16.0]
        response_grid = []

        for theta_deg in orientations:
            theta = np.deg2rad(theta_deg)
            row = []
            for wavelength_px in wavelengths:
                sigma = 0.56 * wavelength_px
                ksize = int(max(9, round(wavelength_px * 6.0))) | 1
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, wavelength_px, 0.5, 0.0, ktype=cv2.CV_32F)
                response = cv2.filter2D(work, cv2.CV_32F, kernel)
                row.append(float(np.mean(np.abs(response))))
            response_grid.append(row)

        response_arr = np.asarray(response_grid, dtype=np.float32)
        orientation_energy = np.mean(response_arr, axis=1)
        wavelength_energy = np.mean(response_arr, axis=0)
        total_energy = float(np.sum(response_arr)) + 1e-9
        dominant_idx = int(np.argmax(response_arr))
        dom_orient_idx, dom_wl_idx = np.unravel_index(dominant_idx, response_arr.shape)
        anisotropy = float((np.max(orientation_energy) - np.mean(orientation_energy)) / (np.mean(orientation_energy) + 1e-9))

        return {
            "orientations_deg": orientations,
            "wavelengths_px": wavelengths,
            "orientation_energy": [float(v) for v in orientation_energy],
            "wavelength_energy": [float(v) for v in wavelength_energy],
            "response_grid": [[float(v) for v in row] for row in response_grid],
            "dominant_orientation_deg": float(orientations[dom_orient_idx]),
            "dominant_wavelength_px": float(wavelengths[dom_wl_idx]),
            "gabor_anisotropy": anisotropy,
            "gabor_energy": total_energy,
        }

    def _analyze_pyramid_texture(self, roi_gray):
        # Update pyramid analysis at a lower rate and reuse cached results
        # on in-between frames to reduce CPU usage.
        update_every = max(1, int(self.pyramid_update_interval))
        if (self.frame_index % update_every) != 0 and self._last_pyramid_data is not None:
            return self._last_pyramid_data

        band_count = self.PYRAMID_BAND_COUNT
        texture_view = self._resize_for_texture_analysis(roi_gray, max_dim=int(max(64, self.pyramid_max_dim)))
        h, w = texture_view.shape
        half_h = h // 2
        half_w = w // 2

        quadrants = {
            "UL": texture_view[0:half_h, 0:half_w],
            "UR": texture_view[0:half_h, half_w:w],
            "LL": texture_view[half_h:h, 0:half_w],
            "LR": texture_view[half_h:h, half_w:w],
        }

        quadrant_bands = {}
        quadrant_raw_bands = {}
        for label, block in quadrants.items():
            if block.size == 0 or min(block.shape[:2]) < 8:
                quadrant_bands[label] = [0.0] * band_count
                quadrant_raw_bands[label] = [0.0] * band_count
            else:
                quadrant_bands[label] = self._pyramid_band_energies(block)
                quadrant_raw_bands[label] = self._pyramid_band_energies(block, normalize=False)

        global_bands = self._pyramid_band_energies(texture_view)
        global_raw_bands = self._pyramid_band_energies(texture_view, normalize=False)
        curr_bands = np.asarray(global_raw_bands, dtype=np.float32)
        _TEMPORAL_GAIN = 350.0
        if self.prev_pyramid_bands is None:
            band_delta = np.zeros_like(curr_bands)
        else:
            band_delta = np.abs(curr_bands - self.prev_pyramid_bands)
        # Raw energies move more with true scene change than normalized ratios.
        # sqrt shape enhances low-level discrimination; 0.5 scale keeps headroom.
        def _t_shape(x):
            return 0.5 * float(np.sqrt(np.clip(x, 0.0, 1.0)))
        temporal_activity_now = _t_shape(_TEMPORAL_GAIN * np.sum(band_delta))
        self.pyramid_temporal_activity = (0.82 * self.pyramid_temporal_activity) + (0.18 * temporal_activity_now)
        self.pyramid_temporal_band_activity = [
            float((0.82 * prev) + (0.18 * _t_shape(_TEMPORAL_GAIN * delta)))
            for prev, delta in zip(self.pyramid_temporal_band_activity, band_delta)
        ]
        self.prev_pyramid_bands = curr_bands

        # Per-quadrant temporal band activity
        quadrant_temporal_bands = {}
        for qlabel, qbands in quadrant_raw_bands.items():
            qcurr = np.asarray(qbands, dtype=np.float32)
            qprev = self.prev_quadrant_bands.get(qlabel)
            if qprev is None:
                qdelta = np.zeros(band_count, dtype=np.float32)
            else:
                qdelta = np.abs(qcurr - qprev)
            prev_qt = self.quadrant_temporal_band_activity.get(qlabel, [0.0] * band_count)
            new_qt = [
                float((0.82 * p) + (0.18 * _t_shape(_TEMPORAL_GAIN * d)))
                for p, d in zip(prev_qt, qdelta)
            ]
            self.quadrant_temporal_band_activity[qlabel] = new_qt
            quadrant_temporal_bands[qlabel] = new_qt
            self.prev_quadrant_bands[qlabel] = qcurr

        def _safe_weighted_centroid(vals):
            arr = np.asarray(vals, dtype=np.float32)
            total = float(np.sum(arr))
            if total <= 1e-9:
                return 0.0
            denom = float(max(1, band_count - 1))
            return float(np.dot(np.arange(band_count, dtype=np.float32), arr) / (total * denom))

        def _renorm01(val, lo, hi):
            span = max(1e-6, float(hi) - float(lo))
            return float(np.clip((float(val) - float(lo)) / span, 0.0, 1.0))

        quadrant_scale_centroids = {}
        quadrant_scale_centroids_renorm = {}
        quadrant_temporal_scale_centroids_raw = {}
        quadrant_temporal_scale_centroids = {}
        for qlabel, qbands in quadrant_bands.items():
            qarr = np.asarray(qbands, dtype=np.float32)
            denom = float(max(1, band_count - 1))
            q_spatial_raw = float(np.dot(np.arange(band_count, dtype=np.float32), qarr) / denom)
            q_temporal_raw = _safe_weighted_centroid(quadrant_temporal_bands.get(qlabel, [0.0] * band_count))
            quadrant_scale_centroids[qlabel] = q_spatial_raw
            quadrant_scale_centroids_renorm[qlabel] = _renorm01(
                q_spatial_raw,
                self.pyramid_centroid_clip["quadrant_spatial"][0],
                self.pyramid_centroid_clip["quadrant_spatial"][1],
            )
            quadrant_temporal_scale_centroids_raw[qlabel] = q_temporal_raw
            quadrant_temporal_scale_centroids[qlabel] = _renorm01(
                q_temporal_raw,
                self.pyramid_centroid_clip["quadrant_temporal"][0],
                self.pyramid_centroid_clip["quadrant_temporal"][1],
            )

        dominant_idx = int(np.argmax(np.asarray(global_bands, dtype=np.float32)))
        centroid = float(np.dot(np.arange(band_count, dtype=np.float32), np.asarray(global_bands, dtype=np.float32))) / float(max(1, band_count - 1))
        centroid_renorm = _renorm01(
            centroid,
            self.pyramid_centroid_clip["global_spatial"][0],
            self.pyramid_centroid_clip["global_spatial"][1],
        )
        temporal_centroid_raw = _safe_weighted_centroid(self.pyramid_temporal_band_activity)
        temporal_centroid = _renorm01(
            temporal_centroid_raw,
            self.pyramid_centroid_clip["global_temporal"][0],
            self.pyramid_centroid_clip["global_temporal"][1],
        )
        band_labels = ["yellow-wide", "coarse", "extra-coarse"]

        out = {
            "global_bands": global_bands,
            "quadrant_bands": quadrant_bands,
            "band_labels": band_labels,
            "dominant_scale_index": dominant_idx,
            "dominant_scale_label": band_labels[dominant_idx],
            "scale_centroid": centroid,
            "scale_centroid_renorm": centroid_renorm,
            "temporal_activity": float(self.pyramid_temporal_activity),
            "temporal_band_activity": list(self.pyramid_temporal_band_activity),
            "temporal_scale_centroid_raw": temporal_centroid_raw,
            "temporal_scale_centroid": temporal_centroid,
            "quadrant_temporal_bands": quadrant_temporal_bands,
            "quadrant_scale_centroids": quadrant_scale_centroids,
            "quadrant_scale_centroids_renorm": quadrant_scale_centroids_renorm,
            "quadrant_temporal_scale_centroids_raw": quadrant_temporal_scale_centroids_raw,
            "quadrant_temporal_scale_centroids": quadrant_temporal_scale_centroids,
        }
        self._last_pyramid_data = out
        return out

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

    def _adapt_flow_interval(self, scale_centroid, wavelength_px):
        """Adapt flow_slow_interval based on prevalent spatial scale.

        Coarser texture / longer wavelength → larger interval so the slow-flow
        baseline spans enough temporal distance to reveal wave-period motion.
        Fine/short-wavelength scenes → smaller interval for faster responsiveness.

        scale_centroid: [0, 1]  0=fine  1=extra-coarse  (from pyramid analysis)
        wavelength_px:  pixels or None
        """
        # Centroid maps 0→2 frames, 1→10 frames (doubled rate vs. previous).
        interval_from_centroid = 2.0 + scale_centroid * 8.0

        if wavelength_px is not None and wavelength_px > 0:
            # Log-map: 10 px→2 frames, 300 px→10 frames
            wl_norm = float(np.clip(
                np.log(max(float(wavelength_px), 1.0) / 10.0) / np.log(300.0 / 10.0),
                0.0, 1.0
            ))
            interval_from_wl = 2.0 + wl_norm * 8.0
            # Blend: wavelength contributes more when it's in the confident long range
            wl_weight = wl_norm * 0.6
            target = (1.0 - wl_weight) * interval_from_centroid + wl_weight * interval_from_wl
        else:
            target = interval_from_centroid

        target = float(np.clip(target, 1.0, 12.0))
        # Slow EMA (~60-frame time constant) to prevent abrupt interval jumps
        self._flow_interval_smooth += 0.016 * (target - self._flow_interval_smooth)
        self.flow_slow_interval = max(1, min(12, int(round(self._flow_interval_smooth))))

    def _adapt_flow_update_interval(self, fast_metrics, temporal_centroid_hz=0.0):
        """Adapt fast-flow update interval to scene dynamics and temporal content."""
        if not self.enable_auto_flow_update_interval:
            return

        activity = float(np.clip(fast_metrics.get("activity", 0.0), 0.0, 1.0))
        speed = float(np.clip(fast_metrics.get("speed_norm", 0.0), 0.0, 1.0))
        coherence = float(np.clip(fast_metrics.get("coherence", 0.0), 0.0, 1.0))

        # Low motion/uncertain scenes keep updates denser for subtle-wave tracking.
        motion = max(activity, speed)
        target = 1.0 + 3.0 * motion
        if coherence < 0.35:
            target -= 0.75

        # Keep update rate above the dominant temporal content when available.
        tc_hz = float(max(0.0, temporal_centroid_hz or 0.0))
        if tc_hz > 0.0 and self.fps > 0.0:
            min_rate_hz = max(2.0, 4.0 * tc_hz)
            max_interval_from_temporal = float(self.fps) / min_rate_hz
            target = min(target, max_interval_from_temporal)

        lo = float(max(1, int(self.flow_update_interval_min)))
        hi = float(max(lo, int(self.flow_update_interval_max)))
        target = float(np.clip(target, lo, hi))

        self._flow_update_interval_smooth += 0.08 * (target - self._flow_update_interval_smooth)
        self.flow_update_interval = max(int(lo), min(int(hi), int(round(self._flow_update_interval_smooth))))

    @staticmethod
    def _flow_directional_region_metrics(mag, ang, region_mask, target_rad, tolerance_deg):
        active = region_mask & (mag > 1e-6)
        if not np.any(active):
            return {
                "support": 0.0,
                "signed_score": 0.0,
                "hit_ratio": 0.0,
                "activity": 0.0,
                "coherence": 0.0,
                "mean_direction_deg": 0.0,
            }

        w = mag[active]
        th = ang[active]
        wsum = float(np.sum(w)) + 1e-6

        cos_delta = np.cos(th - target_rad)
        support = float(np.sum(w * np.maximum(cos_delta, 0.0)) / wsum)
        signed_score = float(np.sum(w * cos_delta) / wsum)

        tol_rad = np.radians(float(max(1.0, tolerance_deg)))
        hits = np.abs(np.angle(np.exp(1j * (th - target_rad)))) <= tol_rad
        hit_ratio = float(np.mean(hits.astype(np.float32)))

        wx = float(np.sum(np.cos(th) * w))
        wy = float(np.sum(np.sin(th) * w))
        coherence = _clip01(float(np.sqrt(wx * wx + wy * wy) / wsum))
        mean_direction = float((np.degrees(np.arctan2(wy, wx)) + 360.0) % 360.0)
        activity = _clip01(float(np.mean(w)) / 8.0)

        return {
            "support": _clip01(support),
            "signed_score": float(np.clip(signed_score, -1.0, 1.0)),
            "hit_ratio": _clip01(hit_ratio),
            "activity": activity,
            "coherence": coherence,
            "mean_direction_deg": mean_direction,
        }

    def _flow_directional_scores(self, flow, target_direction_deg=225.0, tolerance_deg=45.0, stripe_count=5):
        if flow is None:
            return {
                "target_deg": float(target_direction_deg),
                "global": {},
                "quadrants": {},
                "stripes": {},
                "best_quadrant": "-",
                "best_quadrant_support": 0.0,
                "best_stripe": "-",
                "best_stripe_support": 0.0,
            }

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        h, w = mag.shape
        yy, xx = np.mgrid[0:h, 0:w]
        target_rad = np.radians(float(target_direction_deg))

        global_m = self._flow_directional_region_metrics(
            mag,
            ang,
            np.ones((h, w), dtype=bool),
            target_rad,
            tolerance_deg,
        )

        half_h = h // 2
        half_w = w // 2
        quad_masks = {
            "UL": (yy < half_h) & (xx < half_w),
            "UR": (yy < half_h) & (xx >= half_w),
            "LL": (yy >= half_h) & (xx < half_w),
            "LR": (yy >= half_h) & (xx >= half_w),
        }
        quadrants = {
            name: self._flow_directional_region_metrics(mag, ang, qmask, target_rad, tolerance_deg)
            for name, qmask in quad_masks.items()
        }

        stripes = {}
        stripe_count = max(2, int(stripe_count))
        diag_coord = xx.astype(np.float32) + yy.astype(np.float32)
        dmin = float(np.min(diag_coord))
        dmax = float(np.max(diag_coord))
        edges = np.linspace(dmin, dmax, stripe_count + 1)
        for i in range(stripe_count):
            lo = edges[i]
            hi = edges[i + 1]
            smask = (diag_coord >= lo) & ((diag_coord <= hi) if i == (stripe_count - 1) else (diag_coord < hi))
            stripes[f"D{i}"] = self._flow_directional_region_metrics(mag, ang, smask, target_rad, tolerance_deg)

        best_quadrant = "-"
        best_quadrant_support = 0.0
        if quadrants:
            best_quadrant = max(quadrants.keys(), key=lambda k: quadrants[k].get("support", 0.0))
            best_quadrant_support = float(quadrants[best_quadrant].get("support", 0.0))

        best_stripe = "-"
        best_stripe_support = 0.0
        if stripes:
            best_stripe = max(stripes.keys(), key=lambda k: stripes[k].get("support", 0.0))
            best_stripe_support = float(stripes[best_stripe].get("support", 0.0))

        return {
            "target_deg": float(target_direction_deg),
            "global": global_m,
            "quadrants": quadrants,
            "stripes": stripes,
            "best_quadrant": best_quadrant,
            "best_quadrant_support": best_quadrant_support,
            "best_stripe": best_stripe,
            "best_stripe_support": best_stripe_support,
        }

    def _median_filter_direction_deg(self, key, direction_deg):
        """Median filter directional angle in degrees with unwrap to avoid 0/360 jumps."""
        deg = float(direction_deg) % 360.0
        prev = self._flow_dir_unwrapped.get(key)
        unwrapped = deg
        if prev is not None:
            while (unwrapped - prev) > 180.0:
                unwrapped -= 360.0
            while (unwrapped - prev) < -180.0:
                unwrapped += 360.0

        hist = self._flow_dir_history[key]
        hist.append(unwrapped)
        med = float(np.median(np.asarray(hist, dtype=np.float32)))
        self._flow_dir_unwrapped[key] = med
        return med % 360.0

    def _analyze_flow(self, roi_gray, temporal_centroid_hz=0.0):
        """Multi-scale optical flow: fast scale (adjacent frames) + slow scale (N frames apart)."""
        flow_data = dict(self.last_flow_metrics)

        # --- Fast scale: adjacent frames at flow_downscale resolution ---
        fast_w = max(32, int(roi_gray.shape[1] * self.flow_downscale))
        fast_h = max(24, int(roi_gray.shape[0] * self.flow_downscale))
        small_fast = cv2.resize(roi_gray, (fast_w, fast_h), interpolation=cv2.INTER_AREA)

        self.flow_frame_index += 1
        should_update = (self.flow_frame_index % max(1, int(self.flow_update_interval))) == 0
        fast_ready = True

        if self.prev_flow_gray is None:
            self.prev_flow_gray = small_fast
            fast_ready = False
        elif self.prev_flow_gray.shape != small_fast.shape:
            self.prev_flow_gray = small_fast
            self.prev_flow_gray_slow = None
            self._last_slow_metrics = None
            self.last_flow = None
            return flow_data

        fast_m = {
            "activity": float(flow_data.get("fast_activity", 0.0)),
            "speed_norm": float(flow_data.get("fast_speed_norm", 0.0)),
            "direction_deg": float(flow_data.get("fast_direction_deg", 0.0)),
            "coherence": float(flow_data.get("fast_coherence", 0.0)),
        }

        fast_flow_updated = False
        if should_update and fast_ready:
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
            self.last_flow = fast_flow
            fast_m = self._flow_metrics(fast_flow, min_mag=0.25, speed_divisor=6.0, activity_divisor=8.0)
            self._adapt_flow_update_interval(fast_m, temporal_centroid_hz=temporal_centroid_hz)
            fast_flow_updated = True

        self.prev_flow_gray = small_fast

        # --- Slow scale: compare frames flow_slow_interval apart at ~0.15x resolution ---
        # Captures large/slow movements that barely move between adjacent frames.
        is_slow_update = (self.flow_frame_index % max(1, int(self.flow_slow_interval))) == 0

        slow_flow_updated = False
        if is_slow_update or self.prev_flow_gray_slow is None:
            slow_scale = max(0.05, self.flow_downscale * 0.3)
            slow_w = max(16, int(roi_gray.shape[1] * slow_scale))
            slow_h = max(12, int(roi_gray.shape[0] * slow_scale))
            small_slow = cv2.resize(roi_gray, (slow_w, slow_h), interpolation=cv2.INTER_AREA)

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
                    min_mag=0.16 * n,
                    speed_divisor=6.0 * n,
                    activity_divisor=8.0 * n,
                )
                slow_flow_updated = True

            self.prev_flow_gray_slow = small_slow

        # Skip metric updates entirely if no new flow was computed this frame.
        if not fast_flow_updated and not slow_flow_updated:
            return flow_data

        slow_m = self._last_slow_metrics

        # --- Combine fast and slow metrics ---
        # Quality = activity × coherence. The slow scale is only admitted to the
        # direction blend when its coherence is ≥ 0.2; below that the slow flow
        # is considered too incoherent to trust (common in turbulent/pool scenes).
        fq = fast_m["activity"] * fast_m["coherence"]
        slow_usable = (
            slow_m is not None
            and slow_m["activity"] > 0.0
            and slow_m["coherence"] >= 0.2
        )
        sq = (slow_m["activity"] * slow_m["coherence"]) if slow_usable else 0.0
        total_q = fq + sq + 1e-9

        if slow_usable and sq > 0.0:
            # Quality-weighted circular mean for direction.
            fd = np.radians(fast_m["direction_deg"])
            sd = np.radians(slow_m["direction_deg"])
            dx = fq * np.cos(fd) + sq * np.cos(sd)
            dy = fq * np.sin(fd) + sq * np.sin(sd)
            direction = float((np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0)
            speed_norm = _clip01((fq * fast_m["speed_norm"] + sq * slow_m["speed_norm"]) / total_q)
            coherence = _clip01((fq * fast_m["coherence"] + sq * slow_m["coherence"]) / total_q)
            activity = _clip01(max(fast_m["activity"], slow_m["activity"]))
            src_label = "F+S"
        else:
            direction = fast_m["direction_deg"]
            speed_norm = fast_m["speed_norm"]
            coherence = fast_m["coherence"]
            activity = fast_m["activity"]
            sq = 0.0
            src_label = "fast"

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
            "fast_direction_quality": float(fq),
            "slow_direction_quality": float(sq),
            "direction_quality": float(fq + sq),
            "direction_source_label": src_label,
            "adaptive_direction_deg": direction,
            "adaptive_speed_norm": speed_norm,
            "adaptive_coherence": coherence,
            "adaptive_activity": activity,
        })

        # Median-filter directional outputs (size=5) for display/OSC stability.
        flow_data["fast_direction_deg"] = self._median_filter_direction_deg("fast", flow_data["fast_direction_deg"])
        if slow_m is not None and slow_m["activity"] > 0.0:
            flow_data["slow_direction_deg"] = self._median_filter_direction_deg("slow", flow_data["slow_direction_deg"])
        flow_data["direction_deg"] = self._median_filter_direction_deg("blend", flow_data["direction_deg"])
        flow_data["adaptive_direction_deg"] = self._median_filter_direction_deg("adaptive", flow_data["adaptive_direction_deg"])
        self.last_direction = flow_data["direction_deg"]

        if self.flow_axial_mode:
            for _akey in ("fast_direction_deg", "slow_direction_deg", "direction_deg"):
                if _akey in flow_data:
                    flow_data[_akey] = float(flow_data[_akey]) % 180.0

        if self.enable_flow_directional_scoring:
            primary_flow = self.last_flow
            primary_source = "fast"
            if (
                slow_m is not None
                and self.last_flow_slow is not None
                and (slow_m["activity"] * slow_m["coherence"])
                >= (fast_m["activity"] * fast_m["coherence"]) * 0.9
            ):
                primary_flow = self.last_flow_slow
                primary_source = "slow"

            dir_scores = self._flow_directional_scores(
                primary_flow,
                target_direction_deg=self.flow_direction_target_deg,
                tolerance_deg=self.flow_direction_tolerance_deg,
                stripe_count=self.flow_direction_stripe_count,
            )
            flow_data.update({
                "directional_source": primary_source,
                "directional_target_deg": float(self.flow_direction_target_deg),
                "directional_global": dir_scores.get("global", {}),
                "directional_quadrants": dir_scores.get("quadrants", {}),
                "directional_stripes": dir_scores.get("stripes", {}),
                "directional_best_quadrant": dir_scores.get("best_quadrant", "-"),
                "directional_best_quadrant_support": float(dir_scores.get("best_quadrant_support", 0.0)),
                "directional_best_stripe": dir_scores.get("best_stripe", "-"),
                "directional_best_stripe_support": float(dir_scores.get("best_stripe_support", 0.0)),
            })

        # Per-quadrant flow metrics sliced from the current fast/slow flow fields.
        if self.last_flow is not None:
            _fh, _fw = self.last_flow.shape[:2]
            _hh, _hw = _fh // 2, _fw // 2
            flow_data["quadrant_fast_metrics"] = {
                q: self._flow_metrics(sl, min_mag=0.25, speed_divisor=6.0, activity_divisor=8.0)
                for q, sl in {
                    "UL": self.last_flow[0:_hh, 0:_hw],
                    "UR": self.last_flow[0:_hh, _hw:],
                    "LL": self.last_flow[_hh:, 0:_hw],
                    "LR": self.last_flow[_hh:, _hw:],
                }.items()
            }
        if self.last_flow_slow is not None:
            _n = float(self.flow_slow_interval)
            _sfh, _sfw = self.last_flow_slow.shape[:2]
            _shh, _shw = _sfh // 2, _sfw // 2
            flow_data["quadrant_slow_metrics"] = {
                q: self._flow_metrics(sl, min_mag=0.16 * _n, speed_divisor=6.0 * _n, activity_divisor=8.0 * _n)
                for q, sl in {
                    "UL": self.last_flow_slow[0:_shh, 0:_shw],
                    "UR": self.last_flow_slow[0:_shh, _shw:],
                    "LL": self.last_flow_slow[_shh:, 0:_shw],
                    "LR": self.last_flow_slow[_shh:, _shw:],
                }.items()
            }

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

    def _fuse(self, contours, pyramid, flow):
        global_bands = pyramid.get("global_bands", [0.0, 0.0, 0.0])
        dominant_idx = int(pyramid.get("dominant_scale_index", 0))
        raw = {
            # Kept for OSC/UI compatibility; now maps to dominant texture scale.
            "wave_frequency_hz": float(dominant_idx),
            "freq_centroid_hz": float(pyramid.get("scale_centroid", 0.0)),
            "bump_size_common": contours["common"],
            "bump_size_spread": contours["spread"],
            "bump_size_max": contours.get("max", 0.0),
            "bump_size_centroid": contours.get("centroid", 0.0),
            "bump_shape_roundness": contours.get("shape_roundness", 0.0),
            "movement_direction_deg": flow["direction_deg"],
            "movement_speed_norm": flow["speed_norm"],
            "activity": _clip01(0.45 * flow["activity"] + 0.55 * min(1.0, contours["fill_ratio"] * 4.0)),
            "confidence": _clip01(0.4 * contours["stability"] + 0.3 * flow["coherence"] + 0.3 * (1.0 - abs(global_bands[-1] - 0.2))),
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
        mask, roi_gray, blur, thresh, edges, contour_binary, preprocess_display, preprocess_stage_times = self._mask_and_preprocess(gray)
        timers["preprocess_ms"] = (perf_counter() - t0) * 1000.0
        timers.update(preprocess_stage_times)

        analysis_source = thresh if self.enable_threshold_filter else blur

        self.frame_index += 1
        run_full = self.frame_index % self.frame_skip == 0

        # Use combined binary (Canny + Otsu) as contour source by default.
        # When the explicit threshold filter is on, use the Otsu binary image instead.
        contour_source = thresh if self.enable_threshold_filter else contour_binary
        t1 = perf_counter()
        contour_data = self._analyze_contours(contour_source, mask) if run_full else {
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
        pyramid_data = self._analyze_pyramid_texture(analysis_source)
        timers["pyramid_ms"] = (perf_counter() - t2) * 1000.0

        if self.enable_lbp_analysis:
            t_lbp = perf_counter()
            # Compute LBP codes once for the full blur image (the expensive step).
            # Slicing the codes array for each quadrant is zero-copy; _lbp_stats_from_codes
            # (histograms + transitions) is cheap and runs 5× on small sub-arrays.
            _full_codes = WaveAnalyzer._compute_lbp_codes(blur)
            lbp_data = self._lbp_stats_from_codes(_full_codes)
            if _full_codes is not None:
                _ch, _cw = _full_codes.shape[:2]
                _hm, _wm = _ch // 2, _cw // 2
                lbp_data["quadrants"] = {
                    "UL": self._lbp_stats_from_codes(_full_codes[:_hm, :_wm]),
                    "UR": self._lbp_stats_from_codes(_full_codes[:_hm, _wm:]),
                    "LL": self._lbp_stats_from_codes(_full_codes[_hm:, :_wm]),
                    "LR": self._lbp_stats_from_codes(_full_codes[_hm:, _wm:]),
                }
            else:
                lbp_data["quadrants"] = {}
            timers["lbp_ms"] = (perf_counter() - t_lbp) * 1000.0
        else:
            lbp_data = {
                "lbp_mean": 0.0,
                "lbp_std": 0.0,
                "lbp_entropy": 0.0,
                "lbp_uniform_ratio": 0.0,
                "lbp_roughness": 0.0,
                "lbp_dominant_code": 0,
                "lbp_dominant_ratio": 0.0,
                "lbp_smooth": 0.0,
                "lbp_order": 0.0,
                "lbp_chaos": 0.0,
                "lbp_histogram_16": [0.0] * 16,
                "quadrants": {},
            }
            timers["lbp_ms"] = 0.0

        if self.enable_gabor_analysis:
            t_gabor = perf_counter()
            gabor_data = self._analyze_gabor_texture(blur)
            timers["gabor_ms"] = (perf_counter() - t_gabor) * 1000.0
        else:
            gabor_data = {
                "orientations_deg": [0.0, 45.0, 90.0, 135.0],
                "wavelengths_px": [4.0, 8.0, 16.0],
                "orientation_energy": [0.0, 0.0, 0.0, 0.0],
                "wavelength_energy": [0.0, 0.0, 0.0],
                "response_grid": [[0.0, 0.0, 0.0] for _ in range(4)],
                "dominant_orientation_deg": 0.0,
                "dominant_wavelength_px": 0.0,
                "gabor_anisotropy": 0.0,
                "gabor_energy": 0.0,
            }
            timers["gabor_ms"] = 0.0

        # Adapt slow-flow interval to prevalent wave scale before computing flow.
        # Uses self.last_wavelength from the previous frame (one-frame lag, negligible).
        self._adapt_flow_interval(
            pyramid_data.get("scale_centroid", 0.5),
            self.last_wavelength,
        )

        t3 = perf_counter()
        # Optical flow tracks gradients and local texture, so it is more stable
        # on the pre-threshold blurred signal than on a binary image.
        flow_data = self._analyze_flow(
            blur,
            temporal_centroid_hz=pyramid_data.get("temporal_centroid_hz", 0.0),
        )
        timers["flow_ms"] = (perf_counter() - t3) * 1000.0

        t_wl = perf_counter()
        # Wavelength detection: find dominant spatial pattern scale.
        wavelength_data = detect_wavelength(blur, direction="both", lag_range=(3, 300), min_confidence=0.15,
                                             use_median=self.wl_use_median)

        # Per-quadrant wavelength detection uses the same preprocessing settings
        # so local wave scale can be compared against the global estimate.
        h_blur, w_blur = blur.shape[:2]
        h_mid = h_blur // 2
        w_mid = w_blur // 2
        quadrant_rois = {
            "UL": blur[:h_mid, :w_mid],
            "UR": blur[:h_mid, w_mid:],
            "LL": blur[h_mid:, :w_mid],
            "LR": blur[h_mid:, w_mid:],
        }
        quadrant_wavelengths = {}
        for q_name, q_img in quadrant_rois.items():
            if q_img.size == 0 or min(q_img.shape[:2]) < 16:
                quadrant_wavelengths[q_name] = {
                    "wavelength_px": None,
                    "confidence": 0.0,
                    "horizontal_wavelength": None,
                    "vertical_wavelength": None,
                }
                continue
            q_max_lag = max(8, min(300, min(q_img.shape[:2]) // 2))
            quadrant_wavelengths[q_name] = detect_wavelength(
                q_img,
                direction="both",
                lag_range=(3, q_max_lag),
                min_confidence=0.15,
                use_median=self.wl_use_median,
            )
        wavelength_data["quadrants"] = quadrant_wavelengths

        # Median pre-filter (length 5) then 2 Hz IIR lowpass to remove jitter.
        _lp_alpha = float(1.0 - np.exp(-2.0 * np.pi * 2.0 / max(self.fps, 1.0)))
        _raw_wl = wavelength_data.get("wavelength_px")
        if _raw_wl is not None:
            self._wl_history.append(float(_raw_wl))
        if self._wl_history:
            _median_wl = float(np.median(list(self._wl_history)))
            if self._wl_lp_state is None:
                self._wl_lp_state = _median_wl
            else:
                self._wl_lp_state = _lp_alpha * _median_wl + (1.0 - _lp_alpha) * self._wl_lp_state
            wavelength_data["wavelength_px"] = self._wl_lp_state

        # Apply the same median+IIR smoothing per quadrant so displays/OSC use
        # stable local wavelength values rather than raw frame-to-frame estimates.
        for _q in ("UL", "UR", "LL", "LR"):
            _q_item = quadrant_wavelengths.get(_q)
            if not _q_item:
                continue
            _q_raw = _q_item.get("wavelength_px")
            if _q_raw is not None:
                self._wl_q_history[_q].append(float(_q_raw))
            if self._wl_q_history[_q]:
                _q_median = float(np.median(list(self._wl_q_history[_q])))
                if self._wl_q_lp_state[_q] is None:
                    self._wl_q_lp_state[_q] = _q_median
                else:
                    self._wl_q_lp_state[_q] = _lp_alpha * _q_median + (1.0 - _lp_alpha) * self._wl_q_lp_state[_q]
                _q_item["wavelength_px"] = self._wl_q_lp_state[_q]

        self.last_wavelength = wavelength_data["wavelength_px"]
        self.last_wavelength_metrics = dict(wavelength_data)
        timers["wavelength_ms"] = (perf_counter() - t_wl) * 1000.0

        t4 = perf_counter()
        raw, smoothed = self._fuse(contour_data, pyramid_data, flow_data)
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
            "pyramid_data": pyramid_data,
            "lbp_data": lbp_data,
            "gabor_data": gabor_data,
            "flow_data": flow_data,
            "wavelength_data": wavelength_data,
            "raw": raw,
            "smoothed": smoothed,
            "timings": timers,
        }
