import cv2
import numpy as np

_BANDS = [
    (0.1, 0.5, (200, 80, 40), "low"),
    (0.5, 2.0, (60, 160, 60), "mid"),
    (2.0, 5.0, (60, 80, 200), "high"),
]

def render_spectrum_overlay(frame, xf, yf, peaks, show_spectrogram=True, show_summary=True,
                            max_freq=5.0, size=(360, 200), alpha=0.6, margin=12):
    if not show_spectrogram and not show_summary:
        return frame

    h, w = frame.shape[:2]
    box_w, box_h = size
    box_w = min(box_w, max(1, w - margin * 2))
    box_h = min(box_h, max(1, h - margin * 2))

    x0 = max(margin, w - box_w - margin)
    y0 = margin

    panel = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)

    pad_l, pad_r, pad_t, pad_b = 30, 10, 10, 34
    plot_w = max(1, box_w - pad_l - pad_r)
    plot_h = max(1, box_h - pad_t - pad_b)

    if show_spectrogram:
        for fmin, fmax, color, _label in _BANDS:
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

    if show_summary:
        text_y = pad_t + plot_h + 28
        for fmin, fmax, color, label in _BANDS:
            peak = float(peaks.get(label, 0.0)) if peaks else 0.0
            text = f"{label}: {peak:.2f}Hz"
            cv2.putText(panel, text, (8, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
            text_y += 14

    roi = frame[y0:y0 + box_h, x0:x0 + box_w]
    blended = cv2.addWeighted(roi, 1.0 - alpha, panel, alpha, 0)
    frame[y0:y0 + box_h, x0:x0 + box_w] = blended
    return frame