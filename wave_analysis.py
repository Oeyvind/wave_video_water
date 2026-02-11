import numpy as np
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
import cv2

def analyze_direction(prev_gray, gray_small):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_small, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_angle = np.mean(ang[mag > 1.0])
    return float(np.degrees(avg_angle) % 360)

def analyze_frequencies(intensity_series, fps):
    signal = np.array(intensity_series) - np.mean(intensity_series)
    N = len(signal)
    yf = np.abs(rfft(signal))
    xf = rfftfreq(N, 1 / fps)

    def centroid_band(fmin, fmax):
        mask = (xf >= fmin) & (xf <= fmax)
        if np.any(mask):
            band_x = xf[mask]
            band_y = yf[mask]
            total = float(np.sum(band_y))
            if total > 0:
                return float(np.sum(band_x * band_y) / total)
        return 0.0

    return {
        "low": centroid_band(0.1, 0.5),
        "mid": centroid_band(0.5, 2.0),
        "high": centroid_band(2.0, 5.0),
        "xf": xf,
        "yf": yf
    }

def analyze_spatial_frequencies(gray_frame):
    """Analyze spatial frequency (wave crests visible horizontally in frame)"""
    # Sample a horizontal stripe from the middle of the frame
    h, w = gray_frame.shape
    roi = gray_frame[h // 2 : h // 2 + 1, :].flatten().astype(np.float32)
    
    # Remove DC component
    roi_mean = np.mean(roi)
    signal = roi - roi_mean
    
    N = len(signal)
    yf = np.abs(rfft(signal))
    xf = rfftfreq(N, 1.0)  # Normalized spatial frequency (cycles per pixel)
    
    # Replace any NaN values with 0
    yf = np.nan_to_num(yf, nan=0.0, posinf=0.0, neginf=0.0)
    
    def centroid_band(fmin_cycles, fmax_cycles):
        """fmin_cycles and fmax_cycles are in cycles per frame width"""
        fmin_norm = fmin_cycles / w
        fmax_norm = fmax_cycles / w
        mask = (xf >= fmin_norm) & (xf <= fmax_norm)
        if np.any(mask):
            band_x = xf[mask]
            band_y = yf[mask]
            total = float(np.sum(band_y))
            if total > 0:
                return float(np.sum(band_x * band_y) / total * w)  # Convert to cycles per frame width
        return 0.0
    
    return {
        "low": centroid_band(0.5, 3.0),
        "mid": centroid_band(3.0, 8.0),
        "high": centroid_band(8.0, 20.0),
        "xf": xf * w,  # Convert to cycles per frame width
        "yf": yf
    }