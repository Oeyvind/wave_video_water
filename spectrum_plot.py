import cv2
import numpy as np

_BANDS_TEMPORAL = [
    (0.1, 0.5, (200, 80, 40), "low"),
    (0.5, 2.0, (60, 160, 60), "mid"),
    (2.0, 5.0, (60, 80, 200), "high"),
]

_BANDS_SPATIAL = [
    (0.5, 3.0, (200, 80, 40), "low"),
    (3.0, 8.0, (60, 160, 60), "mid"),
    (8.0, 20.0, (60, 80, 200), "high"),
]

def render_spectrum_overlay(frame, xf, yf, peaks, spatial_freqs=None, show_spectrogram=True, show_summary=True,
                            max_freq=5.0, size=(360, 250), alpha=0.6, margin=12):
    # Skip temporal panel if no temporal data provided
    has_temporal_data = len(xf) > 0 and len(yf) > 0
    
    if not has_temporal_data and spatial_freqs is None:
        return frame

    h, w = frame.shape[:2]
    box_w, box_h = size
    box_w = min(box_w, max(1, w - margin * 2))
    box_h = min(box_h, max(1, h - margin * 2))

    x0 = max(margin, w - box_w - margin)
    y0 = margin

    # Draw temporal panel only if we have temporal data
    if has_temporal_data:
        panel = np.zeros((box_h, box_w, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)

        pad_l, pad_r, pad_t, pad_b = 30, 10, 10, 80
        plot_w = max(1, box_w - pad_l - pad_r)
        plot_h = max(1, box_h - pad_t - pad_b)

        if show_spectrogram:
            for fmin, fmax, color, _label in _BANDS_TEMPORAL:
                x1 = pad_l + int((fmin / max_freq) * plot_w)
                x2 = pad_l + int((fmax / max_freq) * plot_w)
                cv2.rectangle(panel, (x1, pad_t), (x2, pad_t + plot_h), color, -1)

            cv2.rectangle(panel, (pad_l, pad_t), (pad_l + plot_w, pad_t + plot_h), (220, 220, 220), 1)

            mask = (xf >= 0.0) & (xf <= max_freq)
            if np.any(mask):
                xf_plot = xf[mask]
                yf_plot = yf[mask]
                max_amp = float(np.max(yf_plot)) if np.max(yf_plot) > 0 else 1.0

                points = []
                for x_val, y_val in zip(xf_plot, yf_plot):
                    x_px = pad_l + int((x_val / max_freq) * plot_w)
                    y_px = pad_t + plot_h - int((y_val / max_amp) * plot_h)
                    points.append((x_px, y_px))
                if len(points) >= 2:
                    cv2.polylines(panel, [np.array(points, dtype=np.int32)], False, (240, 240, 240), 1)

            cv2.putText(panel, "0", (pad_l - 8, pad_t + plot_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
            cv2.putText(panel, f"{max_freq:.1f}Hz", (pad_l + plot_w - 32, pad_t + plot_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
            cv2.putText(panel, "Temporal", (pad_l, pad_t - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        if show_summary:
            text_y = pad_t + plot_h + 28
            for fmin, fmax, color, label in _BANDS_TEMPORAL:
                peak = float(peaks.get(label, 0.0)) if peaks else 0.0
                text = f"{label}: {peak:.2f}Hz"
                cv2.putText(panel, text, (8, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
                text_y += 14

        roi = frame[y0:y0 + box_h, x0:x0 + box_w]
        blended = cv2.addWeighted(roi, 1.0 - alpha, panel, alpha, 0)
        frame[y0:y0 + box_h, x0:x0 + box_w] = blended
        y0 = y0 + box_h + margin
    
    # Spatial frequency panel
    if spatial_freqs is not None:
        spatial_box_h = 250
        
        if y0 + spatial_box_h < h:
            spatial_panel = np.zeros((spatial_box_h, box_w, 3), dtype=np.uint8)
            spatial_panel[:] = (20, 20, 20)
            
            pad_l, pad_r, pad_t, pad_b = 30, 10, 10, 80
            plot_w = max(1, box_w - pad_l - pad_r)
            plot_h = max(1, spatial_box_h - pad_t - pad_b)
            
            max_spatial_freq = 20.0
            if show_spectrogram:
                for fmin, fmax, color, _label in _BANDS_SPATIAL:
                    x1 = pad_l + int((fmin / max_spatial_freq) * plot_w)
                    x2 = pad_l + int((fmax / max_spatial_freq) * plot_w)
                    cv2.rectangle(spatial_panel, (x1, pad_t), (x2, pad_t + plot_h), color, -1)

                cv2.rectangle(spatial_panel, (pad_l, pad_t), (pad_l + plot_w, pad_t + plot_h), (220, 220, 220), 1)

                spatial_xf = spatial_freqs.get("xf", np.array([]))
                spatial_yf = spatial_freqs.get("yf", np.array([]))
                
                # Replace any NaN values
                spatial_xf = np.nan_to_num(spatial_xf, nan=0.0, posinf=0.0, neginf=0.0)
                spatial_yf = np.nan_to_num(spatial_yf, nan=0.0, posinf=0.0, neginf=0.0)
                
                mask = (spatial_xf >= 0.0) & (spatial_xf <= max_spatial_freq)
                if np.any(mask) and len(spatial_xf) > 1:
                    xf_plot = spatial_xf[mask]
                    yf_plot = spatial_yf[mask]
                    
                    # Use log scale for better visibility of low-amplitude signals
                    yf_plot_log = np.log10(np.maximum(yf_plot, 1e-10))
                    max_amp = float(np.max(yf_plot_log)) if len(yf_plot_log) > 0 else 1.0
                    min_amp = float(np.min(yf_plot_log)) if len(yf_plot_log) > 0 else 0.0

                    if max_amp > min_amp:
                        points = []
                        for x_val, y_val_log in zip(xf_plot, yf_plot_log):
                            x_px = pad_l + int((x_val / max_spatial_freq) * plot_w)
                            # Normalize log amplitude to [0, 1] range
                            norm_amp = (y_val_log - min_amp) / (max_amp - min_amp) if max_amp > min_amp else 0.5
                            y_px = pad_t + plot_h - int(norm_amp * plot_h)
                            points.append((x_px, y_px))
                        if len(points) >= 2:
                            cv2.polylines(spatial_panel, [np.array(points, dtype=np.int32)], False, (240, 240, 240), 1)

                cv2.putText(spatial_panel, "0", (pad_l - 8, pad_t + plot_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
                cv2.putText(spatial_panel, f"{max_spatial_freq:.0f}c/f", (pad_l + plot_w - 50, pad_t + plot_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
                cv2.putText(spatial_panel, "Spatial", (pad_l, pad_t - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            if show_summary:
                text_y = pad_t + plot_h + 28
                for fmin, fmax, color, label in _BANDS_SPATIAL:
                    peak = float(spatial_freqs.get(label, 0.0)) if spatial_freqs else 0.0
                    text = f"{label}: {peak:.2f}c/f"
                    cv2.putText(spatial_panel, text, (8, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
                    text_y += 14

            roi = frame[y0:y0 + spatial_box_h, x0:x0 + box_w]
            blended = cv2.addWeighted(roi, 1.0 - alpha, spatial_panel, alpha, 0)
            frame[y0:y0 + spatial_box_h, x0:x0 + box_w] = blended
    
    return frame