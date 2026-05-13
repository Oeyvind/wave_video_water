"""Debug wavelength detector performance."""

import numpy as np
from scipy import signal as scipy_signal

# Generate test patterns
ripple = np.zeros((100, 300), dtype=np.uint8)
for i in range(300):
    ripple[:, i] = (128 + 64 * np.sin(2 * np.pi * i / 10)).astype(np.uint8)

wave = np.zeros((200, 400), dtype=np.uint8)  # Larger image
for i in range(400):
    wave[:, i] = (128 + 64 * np.sin(2 * np.pi * i / 50)).astype(np.uint8)

# Manual autocorrelation on ripple
ripple_row = ripple[50, :].astype(np.float32)
ripple_row = ripple_row - np.mean(ripple_row)
acf_ripple = scipy_signal.correlate(ripple_row, ripple_row, mode='full')[len(ripple_row) - 1:]
acf_ripple = acf_ripple / (acf_ripple[0] + 1e-8)
peaks_r, _ = scipy_signal.find_peaks(acf_ripple[:100], height=0.2, distance=2)
print(f"Ripple ACF peaks at lags: {peaks_r[:5] if len(peaks_r) > 0 else 'none'}")
if len(peaks_r) > 0:
    print(f"  First peak height: {acf_ripple[peaks_r[0]]:.3f}, lag: {peaks_r[0]}")

# Manual autocorrelation on wave
wave_row = wave[100, :].astype(np.float32)
wave_row = wave_row - np.mean(wave_row)
acf_wave = scipy_signal.correlate(wave_row, wave_row, mode='full')[len(wave_row) - 1:]
acf_wave = acf_wave / (acf_wave[0] + 1e-8)
peaks_w, _ = scipy_signal.find_peaks(acf_wave[:200], height=0.2, distance=2)
print(f"Wave ACF peaks at lags: {peaks_w[:5] if len(peaks_w) > 0 else 'none'}")
if len(peaks_w) > 0:
    print(f"  First peak height: {acf_wave[peaks_w[0]]:.3f}, lag: {peaks_w[0]}")

# Now test with the detector
from wavelength_detector import detect_wavelength

result_ripple = detect_wavelength(ripple, direction='horizontal', lag_range=(3, 150), min_confidence=0.1)
print(f"\nDetector on ripple: {result_ripple['wavelength_px']} px, conf {result_ripple['confidence']:.3f}")

result_wave = detect_wavelength(wave, direction='horizontal', lag_range=(5, 250), min_confidence=0.1)
print(f"Detector on wave: {result_wave['wavelength_px']} px, conf {result_wave['confidence']:.3f}")
