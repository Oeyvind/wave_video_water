"""Test wavelength detector with adjusted parameters."""

import numpy as np
from wavelength_detector import detect_wavelength

print("Testing with wider lag range for larger waves...")

# Test 2b: Wave pattern with wider lag range
wave = np.zeros((100, 400), dtype=np.uint8)
for i in range(400):
    wave[:, i] = (128 + 64 * np.sin(2 * np.pi * i / 50)).astype(np.uint8)

result_wave = detect_wavelength(wave, direction='both', lag_range=(5, 200), min_confidence=0.1)
print(f"Wave pattern 50px (wider range): wavelength={result_wave['wavelength_px']}, conf={result_wave['confidence']:.3f}")

# Test 3b: Noise with higher confidence threshold
noise = np.random.randint(50, 200, (100, 300), dtype=np.uint8)
result_noise = detect_wavelength(noise, direction='both', lag_range=(3, 100), min_confidence=0.25)
print(f"Noise (higher conf threshold): wavelength={result_noise['wavelength_px']}, conf={result_noise['confidence']:.3f}")

print("\nAdjustments tested.")
