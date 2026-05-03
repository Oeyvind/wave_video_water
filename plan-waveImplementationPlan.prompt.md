## Plan: Water-Wave Video Analysis Platform

I completed discovery across your current project plus the two reference codebases and built a recommended method stack for realtime use. I could not persist the plan file directly because session memory/file-write is currently unavailable in this environment (no active workspace storage), but the full implementation plan is ready below.

**Recommended approach**
Build a hybrid analysis pipeline that combines:
1. Multi-slit spectral analysis for wave frequency.
2. Contour statistics for bump size distribution.
3. Optical flow (plus optional phase-correlation/autocorr) for direction and speed.
4. Confidence-weighted fusion and smoothing to generate stable control signals for later Csound mapping.

This gives better robustness than any single method while staying within realtime budget on a modern laptop.

**Steps**
1. Define core runtime architecture.
Depends on none.
Create modular stages: capture, mask/ROI, preprocess, analysis engines, signal fusion, overlay renderer, OSC/control output.

2. Implement ROI mask subsystem first.
Depends on 1.
Add quad mask with corner editing workflow: key `k` enters corner-set sequence (UL, UR, LR, LL), key `m` toggles mask display, key `l` saves/loads mask sidecar with same video basename and `.mask` extension.

3. Implement preprocess stack.
Depends on 1 and 2.
Add grayscale normalization, optional background subtraction, optional thresholding, and ROI clipping before all analyses.

4. Implement analysis engine A: contour-based bump metrics.
Depends on 3.
Run edge/threshold pipeline and contour extraction.
Compute per-frame contour area stats:
- mode bin of contour areas (most common bump size)
- median area
- upper quantile area
- contour count and fill ratio
- temporal stability of area distribution

5. Implement analysis engine B: multi-slit frequency analysis.
Depends on 3.
Use 3 to 6 horizontal slits within ROI.
Per slit compute:
- spatial FFT centroid and peak bins
- temporal FFT from ring-buffered slit intensity history
- band energies with low/mid/high mapping
Add Hann windowing and DC removal to reduce edge/leakage artifacts.

6. Implement analysis engine C: direction/speed motion estimation.
Depends on 3.
Primary: dense optical flow at downsampled ROI.
Optional heavier mode: phase correlation/autocorr on ROI patches for displacement vectors.
Compute:
- dominant direction via circular mean
- speed from median flow magnitude or displacement/frame
- coherence/confidence from directional concentration.

7. Implement signal fusion and statistics layer.
Depends on 4, 5, and 6.
Fuse engines into perceptual control signals:
- wave_frequency_hz
- bump_size_common
- bump_size_spread
- movement_direction_deg
- movement_speed_norm
- activity/confidence
Apply smoothing with attack/release or EMA per signal.
Output both raw and smoothed values for debugging.

8. Implement overlay/display mode system.
Depends on 2 to 7.
Create overlay switcher with keyboard toggles for:
- raw frame
- mask-only
- threshold/edge overlay
- contours overlay with area labels
- optical-flow vectors and dominant direction arrow
- FFT/temporal spectra panel
- fused signal HUD
Keep an alpha-blended “processed over original” display option for each stage.

9. Add performance controls and adaptive quality.
Depends on 4 to 8.
Add runtime quality knobs:
- downscale factor
- slit count
- contour min area
- flow resolution
- analysis frame-skip
Target 30 fps realtime with graceful degradation.

10. Add validation workflow and logging.
Depends on 7 to 9.
Record analysis outputs and per-stage timings.
Add diagnostics for confidence drops and mode fallback (for example: contour noise spikes -> trust slit+flow more).

**Relevant files**
- [main.py](main.py) — main loop, input handling, display mode switching, key bindings.
- [wave_analysis.py](wave_analysis.py) — analysis orchestration (extend to multi-engine and fusion).
- [video_capture.py](video_capture.py) — source abstraction for file/live camera, frame pacing hooks.
- [spectrum_plot.py](spectrum_plot.py) — extend for slit/temporal/2D diagnostic plots and unified HUD.

**Reference projects to reuse patterns from**
- `C:\Projects\efx_experiments\wave_video\water` for slit FFT, threshold/correlation, contour experiments.
- `C:\Cabbage_VST\CabbageEfx\midiplugs\domen_ai\Rope` for elaborate display overlays, state extraction, and practical realtime control-signal shaping.

**Verification**
1. Functional checks:
Press `k` to set four corners, `m` to show/hide mask, `l` to save and reload `.mask` sidecar for same video.
2. Analysis checks:
Confirm frequency, bump size, direction, and speed all update meaningfully on test clips with known behavior.
3. Stability checks:
Verify smoothed outputs remain usable for control (no rapid jitter under minor image noise).
4. Performance checks:
Measure per-stage ms/frame and confirm realtime target on laptop with fallback levels available.
5. Visual checks:
Cycle overlay modes and confirm each stage can be inspected on top of original image.

**Decisions**
- Include now:
Hybrid pipeline (contour + slit FFT + flow), mask editing, overlay mode system, realtime performance instrumentation.
- Exclude now:
Audio synthesis/effects mapping logic (next stage as requested).
- Optional advanced mode:
2D FFT/autocorr as a selectable “high quality” analysis mode, not default.

**Further considerations**
1. Use confidence-driven blending between methods (important when one method degrades due lighting/reflections).
2. Keep signal normalization consistent from day one (0..1 or physically scaled units) to simplify later Csound integration.
3. Prefer ring buffers and vectorized NumPy/OpenCV operations to protect realtime performance.

If you want, next I can turn this directly into an implementation task breakdown by milestone (`M1 mask+display`, `M2 core analysis`, `M3 fusion+performance`) with concrete function/class skeletons per file.