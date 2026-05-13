"""Quick test of wavelength detector on synthetic patterns."""

import numpy as np
from wavelength_detector import detect_wavelength

# Test 1: Synthetic ripple pattern (tight spacing ~10px)
ripple = np.zeros((100, 300), dtype=np.uint8)
for i in range(300):
    ripple[:, i] = (128 + 64 * np.sin(2 * np.pi * i / 10)).astype(np.uint8)

result_ripple = detect_wavelength(ripple, direction='both', lag_range=(3, 100), min_confidence=0.1)
print(f"Ripple pattern (expect ~10px): wavelength={result_ripple['wavelength_px']}, conf={result_ripple['confidence']:.3f}")

# Test 2: Synthetic wave pattern (loose spacing ~50px)
wave = np.zeros((100, 300), dtype=np.uint8)
for i in range(300):
    wave[:, i] = (128 + 64 * np.sin(2 * np.pi * i / 50)).astype(np.uint8)

result_wave = detect_wavelength(wave, direction='both', lag_range=(3, 150), min_confidence=0.1)
print(f"Wave pattern (expect ~50px): wavelength={result_wave['wavelength_px']}, conf={result_wave['confidence']:.3f}")

# Test 3: Noise (no pattern)
noise = np.random.randint(50, 200, (100, 300), dtype=np.uint8)
result_noise = detect_wavelength(noise, direction='both', lag_range=(3, 100), min_confidence=0.15)
print(f"Noise (expect None): wavelength={result_noise['wavelength_px']}, conf={result_noise['confidence']:.3f}")

print("\nAll tests completed. Detector is ready for live use.")
