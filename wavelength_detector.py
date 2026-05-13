"""
Autocorrelation-based wavelength detector for ripples and waves.
Detects dominant spatial pattern scale directly from image texture.
"""

import numpy as np
from scipy import signal as scipy_signal


def _prepare_signal(raw, use_median):
    """Apply optional median filter before ACF."""
    sig = raw.astype(np.float32)
    if use_median:
        sig = scipy_signal.medfilt(sig, kernel_size=5)
    sig = sig - np.mean(sig)
    return sig


def detect_wavelength(image, direction="both", lag_range=(3, 300), min_confidence=0.1,
                      use_median=False):
    """
    Detect dominant wavelength by autocorrelation of image rows/columns.
    
    Args:
        image: Grayscale 2D array (blurred image).
        direction: "horizontal", "vertical", or "both" (returns average).
        lag_range: Tuple (min_lag, max_lag) in pixels.
        min_confidence: Minimum peak sharpness to trust detection.
        use_median: Apply median filter (kernel=5) before ACF to suppress glints.
    
    Returns:
        dict with keys:
            "wavelength_px": Dominant wavelength in pixels (or None if not detected).
            "confidence": Peak sharpness metric [0, 1].
            "horizontal_wavelength": H wavelength if direction != "vertical".
            "vertical_wavelength": V wavelength if direction != "horizontal".
    """
    if image is None or image.size == 0:
        return {
            "wavelength_px": None,
            "confidence": 0.0,
            "horizontal_wavelength": None,
            "vertical_wavelength": None,
        }
    
    h, w = image.shape
    min_lag, max_lag = lag_range
    max_lag = min(max_lag, min(h, w) // 2)
    
    h_wavelength = None
    h_confidence = 0.0
    v_wavelength = None
    v_confidence = 0.0
    
    # Horizontal: autocorrelate rows
    if direction in ("horizontal", "both") and w > max_lag:
        # Use several rows for robustness
        row_indices = np.linspace(0, h - 1, min(8, h), dtype=int)
        h_results = []
        
        for ri in row_indices:
            row = _prepare_signal(image[ri, :], use_median)
            if np.std(row) < 1e-6:
                continue
            
            acf = scipy_signal.correlate(row, row, mode="full")[len(row) - 1 :]
            acf = acf / (acf[0] + 1e-8)
            acf = acf[min_lag:max_lag]
            
            if len(acf) > 1:
                # Find first significant peak after initial correlation
                peaks, properties = scipy_signal.find_peaks(
                    acf, height=0.2, distance=2
                )
                if len(peaks) > 0:
                    peak_idx = peaks[0]
                    peak_lag = min_lag + peak_idx
                    peak_height = acf[peak_idx]
                    h_results.append((peak_lag, peak_height))
        
        if h_results:
            # Median wavelength and mean confidence
            wavelengths_h = np.array([r[0] for r in h_results])
            confidences_h = np.array([r[1] for r in h_results])
            h_wavelength = int(np.median(wavelengths_h))
            h_confidence = float(np.mean(confidences_h))
    
    # Vertical: autocorrelate columns
    if direction in ("vertical", "both") and h > max_lag:
        col_indices = np.linspace(0, w - 1, min(8, w), dtype=int)
        v_results = []
        
        for ci in col_indices:
            col = _prepare_signal(image[:, ci], use_median)
            if np.std(col) < 1e-6:
                continue
            
            acf = scipy_signal.correlate(col, col, mode="full")[len(col) - 1 :]
            acf = acf / (acf[0] + 1e-8)
            acf = acf[min_lag:max_lag]
            
            if len(acf) > 1:
                peaks, properties = scipy_signal.find_peaks(
                    acf, height=0.2, distance=2
                )
                if len(peaks) > 0:
                    peak_idx = peaks[0]
                    peak_lag = min_lag + peak_idx
                    peak_height = acf[peak_idx]
                    v_results.append((peak_lag, peak_height))
        
        if v_results:
            wavelengths_v = np.array([r[0] for r in v_results])
            confidences_v = np.array([r[1] for r in v_results])
            v_wavelength = int(np.median(wavelengths_v))
            v_confidence = float(np.mean(confidences_v))
    
    # Combine results
    if direction == "horizontal":
        combined_wl = h_wavelength
        combined_conf = h_confidence
    elif direction == "vertical":
        combined_wl = v_wavelength
        combined_conf = v_confidence
    else:  # both
        if h_wavelength is not None and v_wavelength is not None:
            combined_wl = int(np.mean([h_wavelength, v_wavelength]))
            combined_conf = float(np.mean([h_confidence, v_confidence]))
        elif h_wavelength is not None:
            combined_wl = h_wavelength
            combined_conf = h_confidence
        elif v_wavelength is not None:
            combined_wl = v_wavelength
            combined_conf = v_confidence
        else:
            combined_wl = None
            combined_conf = 0.0
    
    # Only trust if confidence exceeds threshold
    if combined_conf < min_confidence:
        combined_wl = None
    
    return {
        "wavelength_px": combined_wl,
        "confidence": float(combined_conf),
        "horizontal_wavelength": h_wavelength,
        "vertical_wavelength": v_wavelength,
    }
