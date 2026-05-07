import ctypes
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from osc_sender import send_fused_wave_data
from spectrum_plot import render_spectrum_overlay
from video_capture import get_frame
from wave_analysis import WaveAnalyzer


DISPLAY_MODES = [
    "filtered",
    "raw",
]


def focus_window(window_name):
    """Bring the OpenCV window to front and give it real keyboard focus."""
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        # Show a tiny bootstrap frame so the window is materialized before focus.
        bootstrap = np.zeros((120, 200, 3), dtype=np.uint8)
        cv2.putText(bootstrap, "Starting...", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow(window_name, bootstrap)
        cv2.waitKey(1)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
    except cv2.error:
        pass

    # On Windows, SetForegroundWindow silently fails unless the calling process
    # already owns the foreground. The reliable workaround is AttachThreadInput:
    # attach to the foreground window's thread, steal focus, then detach.
    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hwnd = user32.FindWindowW(None, window_name)
        if hwnd:
            fg_hwnd = user32.GetForegroundWindow()
            fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, None)
            our_tid = kernel32.GetCurrentThreadId()
            attached = False
            if fg_tid and fg_tid != our_tid:
                user32.AttachThreadInput(fg_tid, our_tid, True)
                attached = True
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
            user32.SetFocus(hwnd)
            if attached:
                user32.AttachThreadInput(fg_tid, our_tid, False)
    except Exception:
        pass


def draw_transparent_rect(frame, x, y, width, height, alpha=0.45):
    x = max(0, int(x))
    y = max(0, int(y))
    width = max(1, int(width))
    height = max(1, int(height))
    x2 = min(frame.shape[1], x + width)
    y2 = min(frame.shape[0], y + height)
    if x2 <= x or y2 <= y:
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_text_block(
    frame,
    lines,
    anchor_x,
    anchor_y,
    anchor="top_left",
    font_scale=0.5,
    thickness=1,
    color=(235, 235, 235),
    pad=8,
    line_gap=6,
    alpha=0.45,
    right_align=False,
):
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    line_height = max(size[1] for size in text_sizes)
    block_width = max(size[0] for size in text_sizes) + (pad * 2)
    block_height = (pad * 2) + (line_height * len(lines)) + (line_gap * (len(lines) - 1))

    if "right" in anchor:
        x0 = anchor_x - block_width
    else:
        x0 = anchor_x

    if "bottom" in anchor:
        y0 = anchor_y - block_height
    else:
        y0 = anchor_y

    draw_transparent_rect(frame, x0, y0, block_width, block_height, alpha=alpha)

    y = y0 + pad + line_height
    for line, (line_w, _line_h) in zip(lines, text_sizes):
        if right_align:
            x = x0 + block_width - pad - line_w
        else:
            x = x0 + pad
        cv2.putText(frame, line, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height + line_gap

    return x0, y0, block_width, block_height


def measure_text_block(lines, font_scale=0.5, thickness=1, pad=8, line_gap=6):
    if not lines:
        return 0, 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    line_height = max(size[1] for size in text_sizes)
    block_width = max(size[0] for size in text_sizes) + (pad * 2)
    block_height = (pad * 2) + (line_height * len(lines)) + (line_gap * (len(lines) - 1))
    return block_width, block_height


def select_video_source():
    """Select live camera or a file from common wave-video folders."""
    candidates = [
        Path("C:/Projects/efx_experiments/wave_video_files"),
        Path("C:/Cabbage_VST/CabbageEfx/wave_video_files"),
        Path.cwd(),
    ]
    video_dir = None
    for candidate in candidates:
        if candidate.exists():
            video_dir = candidate
            break

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    video_files = []
    if video_dir is not None:
        video_files = sorted(
            [
                f
                for f in os.listdir(video_dir)
                if os.path.isfile(video_dir / f) and Path(f).suffix.lower() in video_extensions
            ]
        )

    print("\n" + "=" * 50)
    print("VIDEO SOURCE SELECTION")
    print("=" * 50)
    print("\n0. Live Camera (Webcam)")
    for idx, video_file in enumerate(video_files, start=1):
        print(f"{idx}. {video_file}")

    print("\n" + "-" * 50)
    while True:
        try:
            choice_num = int(input("Select video source (0 for camera, or enter number): ").strip())
            if choice_num == 0:
                print("\nUsing live camera (index 0)...\n")
                return 0, Path("camera")
            if 1 <= choice_num <= len(video_files):
                selected_file = video_files[choice_num - 1]
                file_path = video_dir / selected_file
                print(f"\nUsing video file: {selected_file}\n")
                return str(file_path), file_path
            print(f"Invalid choice. Please enter 0-{len(video_files)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_mask_path(source_path):
    if source_path.name == "camera":
        return Path.cwd() / "camera.mask"
    return source_path.with_suffix(".mask")


def save_mask(mask_path, points):
    if not points or len(points) != 4:
        return False
    payload = {"points": points}
    mask_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return True


def load_mask(mask_path):
    if not mask_path.exists():
        return None
    try:
        payload = json.loads(mask_path.read_text(encoding="utf-8"))
        points = payload.get("points")
        if isinstance(points, list) and len(points) == 4:
            return [(int(p[0]), int(p[1])) for p in points]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return None


def draw_status_hud(
    frame,
    mode_name,
    quality_idx,
    editing_mask,
    pending_corners,
    analysis,
    perf_stats,
    switches,
    profile_stats,
):
    hud = frame.copy()
    color = (235, 235, 235)

    smooth = analysis["smoothed"]
    timing = analysis["timings"]
    signal_lines = [
        f"freq {smooth['wave_frequency_hz']:.2f} Hz",
        f"bump {smooth['bump_size_common']:.1f} spread {smooth['bump_size_spread']:.1f}",
        f"dir {smooth['movement_direction_deg']:.1f} deg spd {smooth['movement_speed_norm']:.2f}",
        f"act {smooth['activity']:.2f} conf {smooth['confidence']:.2f}",
        f"total {timing['total_ms']:.1f} ms",
    ]

    draw_text_block(
        hud,
        signal_lines,
        10,
        hud.shape[0] - 10,
        anchor="bottom_left",
        font_scale=0.52,
        thickness=1,
        color=color,
    )

    perf_lines = [
        f"source fps: {perf_stats['source_fps']:.2f}",
        f"effective fps: {perf_stats['effective_fps']:.2f}",
        f"processing fps: {perf_stats['processing_fps']:.2f}",
        f"playback ratio: {perf_stats['playback_ratio_pct']:.1f}%",
        f"temp {('on' if switches['temporal_filter'] else 'off')} | screen {('on' if switches['screen_blend'] else 'off')} | blur {('on' if switches['pre_blur'] else 'off')}",
        f"flow detail: {switches['flow_detail_mode']}",
        f"flow: every {switches['flow_interval']} frame(s)",
    ]

    top_cpu = profile_stats.get("top_cpu_stage", "n/a")
    top_cpu_pct = profile_stats.get("top_cpu_pct", 0.0)
    perf_lines.append(f"top cpu: {top_cpu} ({top_cpu_pct:.1f}%)")
    perf_w, perf_h = measure_text_block(perf_lines, font_scale=0.48, thickness=1, pad=8, line_gap=6)

    profile_lines = [
        "CPU Stage Profile (avg ms | share)",
        f"capture  {profile_stats.get('capture_ms', 0.0):.2f} | {profile_stats.get('capture_pct', 0.0):.1f}%",
        f"preproc  {profile_stats.get('preprocess_ms', 0.0):.2f} | {profile_stats.get('preprocess_pct', 0.0):.1f}%",
        f"temporal {profile_stats.get('temporal_filter_ms', 0.0):.2f} | {profile_stats.get('temporal_filter_pct', 0.0):.1f}%",
        f"tdiff    {profile_stats.get('temporal_diff_ms', 0.0):.2f} | {profile_stats.get('temporal_diff_pct', 0.0):.1f}%",
        f"screen   {profile_stats.get('screen_blend_ms', 0.0):.2f} | {profile_stats.get('screen_blend_pct', 0.0):.1f}%",
        f"blur     {profile_stats.get('pre_blur_ms', 0.0):.2f} | {profile_stats.get('pre_blur_pct', 0.0):.1f}%",
        f"contours {profile_stats.get('contours_ms', 0.0):.2f} | {profile_stats.get('contours_pct', 0.0):.1f}%",
        f"slits    {profile_stats.get('slits_ms', 0.0):.2f} | {profile_stats.get('slits_pct', 0.0):.1f}%",
        f"flow     {profile_stats.get('flow_ms', 0.0):.2f} | {profile_stats.get('flow_pct', 0.0):.1f}%",
        f"fusion   {profile_stats.get('fusion_ms', 0.0):.2f} | {profile_stats.get('fusion_pct', 0.0):.1f}%",
        f"render   {profile_stats.get('render_ms', 0.0):.2f} | {profile_stats.get('render_pct', 0.0):.1f}%",
        f"osc      {profile_stats.get('osc_ms', 0.0):.2f} | {profile_stats.get('osc_pct', 0.0):.1f}%",
        f"waitKey  {profile_stats.get('wait_ms', 0.0):.2f} | {profile_stats.get('wait_pct', 0.0):.1f}%",
        f"loop     {profile_stats.get('loop_ms', 0.0):.2f} | 100.0%",
    ]
    profile_w, profile_h = measure_text_block(profile_lines, font_scale=0.42, thickness=1, pad=8, line_gap=6)

    legend_lines = [
        "Keys",
        f"[D] mode: {mode_name}",
        f"[1..4] quality: {quality_idx + 1}",
        f"[T] temporal: {'on' if switches['temporal_filter'] else 'off'}",
        f"[H] temporal diff: {'on' if switches['temporal_diff_filter'] else 'off'}",
        f"[G] tdiff polarity: {switches['temporal_diff_polarity']}",
        f"[E] screen: {'on' if switches['screen_blend'] else 'off'}",
        f"[B] blur: {'on' if switches['pre_blur'] else 'off'}",
        f"[R] threshold overlay: {'on' if switches['threshold_overlay'] else 'off'}",
        f"[C] contour overlay: {'on' if switches['contour_overlay'] else 'off'}",
        f"[S] spectrum overlay (tap@blur): {'on' if switches['spectrum_overlay'] else 'off'}",
        f"[X] spectrum view: {switches['spectrum_view_mode']}",
        f"[Z] spectral mode: {switches['spectral_mode']}",
        f"[V] flow overlay (tap@blur): {'on' if switches['flow_overlay'] else 'off'}",
        f"[F] flow detail: {switches['flow_detail_mode']}",
        f"flow cadence: 1/{switches['flow_interval']}",
        f"[M] show mask: {'on' if switches['show_mask'] else 'off'}",
        f"[K] mask edit: {'on' if switches['editing_mask'] else 'off'}",
        f"[N] step mode: {'on' if switches['step_mode'] else 'off'}",
        "[L] load mask",
        "[Shift+L] save mask",
        "[P] pause  [Q] quit",
    ]
    legend_box = draw_text_block(
        hud,
        legend_lines,
        hud.shape[1] - 10,
        10,
        anchor="top_right",
        font_scale=0.48,
        thickness=1,
        color=color,
        right_align=True,
    )

    legend_bottom = legend_box[1] + legend_box[3]
    perf_top_y = hud.shape[0] - perf_h - 10
    profile_top_y = min(legend_bottom + 8, max(legend_bottom + 8, perf_top_y - profile_h - 8))

    draw_text_block(
        hud,
        profile_lines,
        hud.shape[1] - 10,
        profile_top_y,
        anchor="top_right",
        font_scale=0.42,
        thickness=1,
        color=color,
        right_align=True,
    )

    draw_text_block(
        hud,
        perf_lines,
        hud.shape[1] - 10,
        hud.shape[0] - 10,
        anchor="bottom_right",
        font_scale=0.48,
        thickness=1,
        color=color,
        right_align=True,
    )

    return hud


def draw_flow_overlay(frame, flow):
    if flow is None:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    fh, fw = flow.shape[:2]
    # Sample every 2nd point in both X and Y relative to previous density.
    # This halves visible point density per axis (quarter total vectors).
    step = 16
    sx = w / float(fw)
    sy = h / float(fh)
    line_thickness = 3

    for y in range(0, fh, step):
        for x in range(0, fw, step):
            dx, dy = flow[y, x]
            p0 = (int(x * sx), int(y * sy))
            p1 = (int((x + dx * 2.0) * sx), int((y + dy * 2.0) * sy))
            cv2.line(out, p0, p1, (30, 180, 255), line_thickness)
    return out


def _quadrant_vector_stats(flow_block):
    if flow_block.size == 0:
        return 0.0, 0.0, False

    dx = flow_block[..., 0]
    dy = flow_block[..., 1]
    mag = np.sqrt(dx * dx + dy * dy)
    active = mag > 0.12
    if not np.any(active):
        return 0.0, 0.0, False

    weights = mag[active]
    sum_x = float(np.sum(dx[active] * weights))
    sum_y = float(np.sum(dy[active] * weights))

    direction_deg = (np.degrees(np.arctan2(sum_y, sum_x)) + 360.0) % 360.0
    strength = float(np.median(weights))
    return direction_deg, strength, True


def draw_quadrant_flow_arrows(frame, flow):
    if flow is None:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]
    fh, fw = flow.shape[:2]
    half_h = fh // 2
    half_w = fw // 2

    quadrants = [
        ("UL", flow[0:half_h, 0:half_w], (int(w * 0.25), int(h * 0.25))),
        ("UR", flow[0:half_h, half_w:fw], (int(w * 0.75), int(h * 0.25))),
        ("LL", flow[half_h:fh, 0:half_w], (int(w * 0.25), int(h * 0.75))),
        ("LR", flow[half_h:fh, half_w:fw], (int(w * 0.75), int(h * 0.75))),
    ]

    for label, block, center in quadrants:
        direction_deg, strength, has_motion = _quadrant_vector_stats(block)
        radius = 34
        cv2.circle(out, center, radius, (220, 220, 220), 1)

        if has_motion:
            arrow_len = int(np.clip(16 + strength * 8.0, 16, 44))
            dx = int(np.cos(np.radians(direction_deg)) * arrow_len)
            dy = int(np.sin(np.radians(direction_deg)) * arrow_len)
            cv2.arrowedLine(out, center, (center[0] + dx, center[1] + dy), (80, 190, 255), 2, tipLength=0.22)

        cv2.putText(out, label, (center[0] - 12, center[1] - radius - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)

    # Global summary arrow (all quadrants combined) for local-vs-global comparison.
    g_direction_deg, g_strength, g_has_motion = _quadrant_vector_stats(flow)
    g_center = (int(w * 0.5), int(h * 0.5))
    g_radius = 42
    cv2.circle(out, g_center, g_radius, (240, 240, 240), 1)
    if g_has_motion:
        g_len = int(np.clip(20 + g_strength * 10.0, 20, 56))
        g_dx = int(np.cos(np.radians(g_direction_deg)) * g_len)
        g_dy = int(np.sin(np.radians(g_direction_deg)) * g_len)
        cv2.arrowedLine(out, g_center, (g_center[0] + g_dx, g_center[1] + g_dy), (120, 220, 255), 3, tipLength=0.22)
    cv2.putText(out, "G", (g_center[0] - 7, g_center[1] - g_radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)

    return out


def draw_spectral_slits_overlay(frame, slit_rows):
    if not slit_rows:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]
    for idx, row in enumerate(slit_rows):
        y = int(np.clip(row, 0, h - 1))
        cv2.line(out, (0, y), (w - 1, y), (255, 210, 60), 1)
        if idx == 0:
            cv2.putText(out, "spectral slits", (10, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 220, 120), 1, cv2.LINE_AA)
    return out


def make_display_frame(
    base_frame,
    analysis,
    mode_name,
    analyzer,
    threshold_overlay,
    contour_overlay,
    flow_overlay,
    spectrum_overlay,
    spectrum_view_mode,
):
    if mode_name == "filtered":
        # Show the exact grayscale signal used for threshold/contour/spectral/flow taps.
        out = cv2.cvtColor(analysis["preprocess_display"], cv2.COLOR_GRAY2BGR)
    else:
        out = base_frame.copy()

    if threshold_overlay:
        thr_rgb = cv2.cvtColor(analysis["threshold"], cv2.COLOR_GRAY2BGR)
        edge_rgb = cv2.cvtColor(analysis["edges"], cv2.COLOR_GRAY2BGR)
        merged = cv2.addWeighted(thr_rgb, 0.7, edge_rgb, 0.3, 0)
        # Keep a faint view of the base signal, but make binary threshold dominant.
        out = cv2.addWeighted(out, 0.2, merged, 0.8, 0)

    if contour_overlay:
        cv2.drawContours(out, analysis["contours"], -1, (30, 250, 70), 2)

    if flow_overlay:
        out = draw_flow_overlay(out, analyzer.last_flow)
        out = draw_quadrant_flow_arrows(out, analyzer.last_flow)

    if spectrum_overlay:
        slit = analysis["slit_data"]
        if analyzer.spectral_mode == "slits":
            out = draw_spectral_slits_overlay(out, slit.get("slit_rows", []))
        spatial_freqs = {
            "low": slit["band_low"],
            "mid": slit["band_mid"],
            "high": slit["band_high"],
            "peak_common": slit.get("spatial_peak_common", 0.0),
            "xf": slit.get("spatial_xf", np.array([])),
            "yf": slit.get("spatial_yf", np.array([])),
        }
        render_spectrum_overlay(
            out,
            slit["temporal_xf"],
            slit["temporal_yf"],
            {
                "low": analysis["smoothed"]["wave_frequency_hz"] * 0.5,
                "mid": analysis["smoothed"]["wave_frequency_hz"],
                "high": analysis["smoothed"]["wave_frequency_hz"] * 1.5,
            },
            spatial_freqs=spatial_freqs,
            quadrant_summaries=slit.get("quadrant_summaries", None),
            layout_mode=spectrum_view_mode,
            show_spectrogram=True,
            show_summary=True,
            side="left",
        )

    return out


def main():
    video_source, source_path = select_video_source()
    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video FPS: {fps:.2f}")

    analyzer = WaveAnalyzer(fps=fps)
    quality_presets = [
        {"downscale": 1.0, "slit_count": 6, "contour_min_area": 20.0, "frame_skip": 1},
        {"downscale": 0.75, "slit_count": 5, "contour_min_area": 28.0, "frame_skip": 1},
        {"downscale": 0.6, "slit_count": 4, "contour_min_area": 36.0, "frame_skip": 1},
        {"downscale": 0.45, "slit_count": 3, "contour_min_area": 45.0, "frame_skip": 2},
    ]
    flow_quality_presets_high = [
        {
            "flow_downscale": 0.55,
            "flow_update_interval": 1,
            "flow_pyr_scale": 0.5,
            "flow_levels": 3,
            "flow_winsize": 15,
            "flow_iterations": 3,
            "flow_poly_n": 5,
            "flow_poly_sigma": 1.2,
            "flow_flags": 0,
        },
        {
            "flow_downscale": 0.5,
            "flow_update_interval": 1,
            "flow_pyr_scale": 0.5,
            "flow_levels": 3,
            "flow_winsize": 13,
            "flow_iterations": 2,
            "flow_poly_n": 5,
            "flow_poly_sigma": 1.1,
            "flow_flags": 0,
        },
        {
            "flow_downscale": 0.4,
            "flow_update_interval": 2,
            "flow_pyr_scale": 0.5,
            "flow_levels": 2,
            "flow_winsize": 11,
            "flow_iterations": 2,
            "flow_poly_n": 5,
            "flow_poly_sigma": 1.1,
            "flow_flags": 0,
        },
        {
            "flow_downscale": 0.33,
            "flow_update_interval": 3,
            "flow_pyr_scale": 0.5,
            "flow_levels": 2,
            "flow_winsize": 9,
            "flow_iterations": 1,
            "flow_poly_n": 5,
            "flow_poly_sigma": 1.1,
            "flow_flags": 0,
        },
    ]

    flow_quality_presets_reduced = [
        {
            "flow_downscale": 0.275,
            "flow_update_interval": 1,
            "flow_pyr_scale": 0.5,
            "flow_levels": 3,
            "flow_winsize": 15,
            "flow_iterations": 3,
            "flow_poly_n": 5,
            "flow_poly_sigma": 1.2,
            "flow_flags": 0,
        },
        {
            "flow_downscale": 0.25,
            "flow_update_interval": 2,
            "flow_pyr_scale": 0.5,
            "flow_levels": 3,
            "flow_winsize": 13,
            "flow_iterations": 2,
            "flow_poly_n": 5,
            "flow_poly_sigma": 1.1,
            "flow_flags": 0,
        },
        {
            "flow_downscale": 0.2,
            "flow_update_interval": 3,
            "flow_pyr_scale": 0.5,
            "flow_levels": 2,
            "flow_winsize": 11,
            "flow_iterations": 2,
            "flow_poly_n": 5,
            "flow_poly_sigma": 1.1,
            "flow_flags": 0,
        },
        {
            "flow_downscale": 0.165,
            "flow_update_interval": 4,
            "flow_pyr_scale": 0.5,
            "flow_levels": 2,
            "flow_winsize": 9,
            "flow_iterations": 1,
            "flow_poly_n": 5,
            "flow_poly_sigma": 1.1,
            "flow_flags": 0,
        },
    ]

    flow_quality_presets_lowest = [
        {
            "flow_downscale": 0.1375,
            "flow_update_interval": 2,
            "flow_pyr_scale": 0.5,
            "flow_levels": 3,
            "flow_winsize": 15,
            "flow_iterations": 3,
            "flow_poly_n": 5,
            "flow_poly_sigma": 1.2,
            "flow_flags": 0,
        },
        {
            "flow_downscale": 0.125,
            "flow_update_interval": 3,
            "flow_pyr_scale": 0.5,
            "flow_levels": 3,
            "flow_winsize": 13,
            "flow_iterations": 2,
            "flow_poly_n": 5,
            "flow_poly_sigma": 1.1,
            "flow_flags": 0,
        },
        {
            "flow_downscale": 0.1,
            "flow_update_interval": 4,
            "flow_pyr_scale": 0.5,
            "flow_levels": 2,
            "flow_winsize": 11,
            "flow_iterations": 2,
            "flow_poly_n": 5,
            "flow_poly_sigma": 1.1,
            "flow_flags": 0,
        },
        {
            "flow_downscale": 0.0825,
            "flow_update_interval": 6,
            "flow_pyr_scale": 0.5,
            "flow_levels": 2,
            "flow_winsize": 9,
            "flow_iterations": 1,
            "flow_poly_n": 5,
            "flow_poly_sigma": 1.1,
            "flow_flags": 0,
        },
    ]

    flow_detail_modes = ["high", "reduced", "lowest"]
    flow_detail_mode = "lowest"

    def apply_flow_quality_for_current_mode(idx):
        if flow_detail_mode == "high":
            analyzer.set_flow_quality(**flow_quality_presets_high[idx])
        elif flow_detail_mode == "reduced":
            analyzer.set_flow_quality(**flow_quality_presets_reduced[idx])
        else:
            analyzer.set_flow_quality(**flow_quality_presets_lowest[idx])

    quality_idx = 2
    analyzer.set_quality(**quality_presets[quality_idx])
    apply_flow_quality_for_current_mode(quality_idx)
    loop_video = not isinstance(video_source, int)
    if loop_video:
        print("Video looping is enabled for file playback.")

    mask_path = get_mask_path(source_path)
    loaded_points = load_mask(mask_path)
    if loaded_points:
        analyzer.set_mask_points(loaded_points)
        print(f"Loaded mask from {mask_path}")

    mode_idx = 0
    step = False
    show_flow_overlay = True
    show_contour_overlay = False
    show_threshold_overlay = False
    show_spectrum_overlay = True
    spectrum_view_modes = ["full", "quadrants"]
    spectrum_view_mode = "full"
    editing_mask = False
    pending_corners = ["UL", "UR", "LR", "LL"]
    corner_points = []
    last_analysis = None
    frame_timestamps = deque(maxlen=160)
    processing_fps_values = deque(maxlen=160)
    profile_keys = [
        "capture_ms",
        "preprocess_ms",
        "temporal_filter_ms",
        "temporal_diff_ms",
        "screen_blend_ms",
        "pre_blur_ms",
        "contours_ms",
        "slits_ms",
        "flow_ms",
        "fusion_ms",
        "render_ms",
        "osc_ms",
        "wait_ms",
        "loop_ms",
    ]
    profile_windows = {k: deque(maxlen=160) for k in profile_keys}
    current_profile_stats = {
        "capture_ms": 0.0,
        "preprocess_ms": 0.0,
        "temporal_filter_ms": 0.0,
        "temporal_diff_ms": 0.0,
        "screen_blend_ms": 0.0,
        "pre_blur_ms": 0.0,
        "contours_ms": 0.0,
        "slits_ms": 0.0,
        "flow_ms": 0.0,
        "fusion_ms": 0.0,
        "render_ms": 0.0,
        "osc_ms": 0.0,
        "wait_ms": 0.0,
        "loop_ms": 0.0,
        "capture_pct": 0.0,
        "preprocess_pct": 0.0,
        "temporal_filter_pct": 0.0,
        "temporal_diff_pct": 0.0,
        "screen_blend_pct": 0.0,
        "pre_blur_pct": 0.0,
        "contours_pct": 0.0,
        "slits_pct": 0.0,
        "flow_pct": 0.0,
        "fusion_pct": 0.0,
        "render_pct": 0.0,
        "osc_pct": 0.0,
        "wait_pct": 0.0,
        "top_cpu_stage": "collecting",
        "top_cpu_pct": 0.0,
    }

    def on_mouse(event, x, y, _flags, _param):
        nonlocal editing_mask, pending_corners, corner_points
        if not editing_mask:
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        corner_points.append((int(x), int(y)))
        if pending_corners:
            pending_corners.pop(0)
        if len(corner_points) == 4:
            analyzer.set_mask_points(corner_points)
            editing_mask = False
            pending_corners = ["UL", "UR", "LR", "LL"]
            print("Mask corners updated")

    cv2.namedWindow("Wave Analyzer")
    cv2.setMouseCallback("Wave Analyzer", on_mouse)
    focus_window("Wave Analyzer")

    while True:
        loop_start = time.perf_counter()
        cap_t0 = time.perf_counter()
        result = get_frame(cap, loop=loop_video)
        cap_t1 = time.perf_counter()
        profile_windows["capture_ms"].append((cap_t1 - cap_t0) * 1000.0)
        if result is None:
            break

        frame, gray = result
        frame_timestamps.append(time.perf_counter())
        analysis = analyzer.analyze(gray)
        last_analysis = analysis

        total_ms = max(analysis["timings"].get("total_ms", 0.0), 1e-6)
        processing_fps_values.append(1000.0 / total_ms)

        effective_fps = 0.0
        if len(frame_timestamps) >= 2:
            elapsed = frame_timestamps[-1] - frame_timestamps[0]
            if elapsed > 0:
                effective_fps = (len(frame_timestamps) - 1) / elapsed

        processing_fps = float(np.mean(processing_fps_values)) if processing_fps_values else 0.0
        playback_ratio_pct = (effective_fps / fps * 100.0) if fps > 0 else 0.0

        perf_stats = {
            "source_fps": float(fps),
            "effective_fps": float(effective_fps),
            "processing_fps": float(processing_fps),
            "playback_ratio_pct": float(playback_ratio_pct),
        }

        switches = {
            "show_mask": analyzer.show_mask,
            "editing_mask": editing_mask,
            "step_mode": step,
            "flow_interval": analyzer.flow_update_interval,
            "flow_detail_mode": flow_detail_mode,
            "temporal_diff_filter": analyzer.enable_temporal_difference_filter,
            "temporal_diff_polarity": analyzer.temporal_diff_polarity,
            "spectral_mode": analyzer.spectral_mode,
            "flow_overlay": show_flow_overlay,
            "contour_overlay": show_contour_overlay,
            "threshold_overlay": show_threshold_overlay,
            "spectrum_overlay": show_spectrum_overlay,
            "spectrum_view_mode": spectrum_view_mode,
            "temporal_filter": analyzer.enable_temporal_change_filter,
            "screen_blend": analyzer.enable_screen_blend_equalization,
            "pre_blur": analyzer.enable_preprocess_blur,
        }

        mode_name = DISPLAY_MODES[mode_idx]
        render_t0 = time.perf_counter()
        display = make_display_frame(
            frame,
            analysis,
            mode_name,
            analyzer,
            show_threshold_overlay,
            show_contour_overlay,
            show_flow_overlay,
            show_spectrum_overlay,
            spectrum_view_mode,
        )
        display = draw_status_hud(
            display,
            mode_name,
            quality_idx,
            editing_mask,
            pending_corners,
            analysis,
            perf_stats,
            switches,
            current_profile_stats,
        )

        if analyzer.show_mask and analyzer.mask_points:
            pts = np.array(analyzer.mask_points, dtype=np.int32)
            cv2.polylines(display, [pts], True, (255, 210, 80), 2)
            for p in analyzer.mask_points:
                cv2.circle(display, p, 4, (255, 210, 80), -1)

        if editing_mask and corner_points:
            for point in corner_points:
                cv2.circle(display, point, 5, (20, 200, 250), -1)

        render_t1 = time.perf_counter()
        profile_windows["render_ms"].append((render_t1 - render_t0) * 1000.0)

        cv2.imshow("Wave Analyzer", display)

        if last_analysis is not None:
            osc_t0 = time.perf_counter()
            send_fused_wave_data(last_analysis)
            osc_t1 = time.perf_counter()
            profile_windows["osc_ms"].append((osc_t1 - osc_t0) * 1000.0)

        frame_time = max(1, int(1000 / fps))
        wait_t0 = time.perf_counter()
        key = cv2.waitKey(frame_time) & 0xFF
        wait_t1 = time.perf_counter()
        profile_windows["wait_ms"].append((wait_t1 - wait_t0) * 1000.0)

        profile_windows["preprocess_ms"].append(analysis["timings"].get("preprocess_ms", 0.0))
        profile_windows["temporal_filter_ms"].append(analysis["timings"].get("temporal_filter_ms", 0.0))
        profile_windows["temporal_diff_ms"].append(analysis["timings"].get("temporal_diff_ms", 0.0))
        profile_windows["screen_blend_ms"].append(analysis["timings"].get("screen_blend_ms", 0.0))
        profile_windows["pre_blur_ms"].append(analysis["timings"].get("pre_blur_ms", 0.0))
        profile_windows["contours_ms"].append(analysis["timings"].get("contours_ms", 0.0))
        profile_windows["slits_ms"].append(analysis["timings"].get("slits_ms", 0.0))
        profile_windows["flow_ms"].append(analysis["timings"].get("flow_ms", 0.0))
        profile_windows["fusion_ms"].append(analysis["timings"].get("fusion_ms", 0.0))
        loop_end = time.perf_counter()
        profile_windows["loop_ms"].append((loop_end - loop_start) * 1000.0)

        avg_profile_ms = {k: (float(np.mean(v)) if v else 0.0) for k, v in profile_windows.items()}
        total_for_pct = max(avg_profile_ms["loop_ms"], 1e-6)
        avg_profile_pct = {
            "capture_pct": avg_profile_ms["capture_ms"] / total_for_pct * 100.0,
            "preprocess_pct": avg_profile_ms["preprocess_ms"] / total_for_pct * 100.0,
            "temporal_filter_pct": avg_profile_ms["temporal_filter_ms"] / total_for_pct * 100.0,
            "temporal_diff_pct": avg_profile_ms["temporal_diff_ms"] / total_for_pct * 100.0,
            "screen_blend_pct": avg_profile_ms["screen_blend_ms"] / total_for_pct * 100.0,
            "pre_blur_pct": avg_profile_ms["pre_blur_ms"] / total_for_pct * 100.0,
            "contours_pct": avg_profile_ms["contours_ms"] / total_for_pct * 100.0,
            "slits_pct": avg_profile_ms["slits_ms"] / total_for_pct * 100.0,
            "flow_pct": avg_profile_ms["flow_ms"] / total_for_pct * 100.0,
            "fusion_pct": avg_profile_ms["fusion_ms"] / total_for_pct * 100.0,
            "render_pct": avg_profile_ms["render_ms"] / total_for_pct * 100.0,
            "osc_pct": avg_profile_ms["osc_ms"] / total_for_pct * 100.0,
            "wait_pct": avg_profile_ms["wait_ms"] / total_for_pct * 100.0,
        }

        ranked = sorted(
            [
                ("capture", avg_profile_pct["capture_pct"]),
                ("preproc", avg_profile_pct["preprocess_pct"]),
                ("temporal", avg_profile_pct["temporal_filter_pct"]),
                ("tdiff", avg_profile_pct["temporal_diff_pct"]),
                ("screen", avg_profile_pct["screen_blend_pct"]),
                ("blur", avg_profile_pct["pre_blur_pct"]),
                ("contours", avg_profile_pct["contours_pct"]),
                ("slits", avg_profile_pct["slits_pct"]),
                ("flow", avg_profile_pct["flow_pct"]),
                ("fusion", avg_profile_pct["fusion_pct"]),
                ("render", avg_profile_pct["render_pct"]),
                ("osc", avg_profile_pct["osc_pct"]),
                ("waitKey", avg_profile_pct["wait_pct"]),
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        top_cpu_stage, top_cpu_pct = ranked[0]

        current_profile_stats = {
            **avg_profile_ms,
            **avg_profile_pct,
            "top_cpu_stage": top_cpu_stage,
            "top_cpu_pct": top_cpu_pct,
        }

        if key == ord("q"):
            break
        if key == ord("p"):
            cv2.waitKey(-1)
        if key == ord("n"):
            step = not step
        if key == ord("d"):
            mode_idx = (mode_idx + 1) % len(DISPLAY_MODES)
        if key == ord("v"):
            show_flow_overlay = not show_flow_overlay
        if key == ord("c"):
            show_contour_overlay = not show_contour_overlay
        if key == ord("r"):
            show_threshold_overlay = not show_threshold_overlay
        if key == ord("s"):
            show_spectrum_overlay = not show_spectrum_overlay
        if key == ord("x"):
            next_idx = (spectrum_view_modes.index(spectrum_view_mode) + 1) % len(spectrum_view_modes)
            spectrum_view_mode = spectrum_view_modes[next_idx]
        if key == ord("z"):
            spectral_modes = ["slits", "fft2"]
            next_idx = (spectral_modes.index(analyzer.spectral_mode) + 1) % len(spectral_modes)
            analyzer.spectral_mode = spectral_modes[next_idx]
            analyzer.last_slit_data = None
            analyzer.global_history.clear()
            analyzer.slit_history = [deque(maxlen=buf.maxlen) for buf in analyzer.slit_history]
        if key == ord("m"):
            analyzer.show_mask = not analyzer.show_mask
        if key == ord("f"):
            next_idx = (flow_detail_modes.index(flow_detail_mode) + 1) % len(flow_detail_modes)
            flow_detail_mode = flow_detail_modes[next_idx]
            apply_flow_quality_for_current_mode(quality_idx)
        if key == ord("t"):
            analyzer.enable_temporal_change_filter = not analyzer.enable_temporal_change_filter
            analyzer.prev_temporal_lpf_float = None
            analyzer.prev_temporal_fast_float = None
            if not analyzer.enable_temporal_change_filter:
                profile_windows["temporal_filter_ms"].clear()
        if key == ord("h"):
            analyzer.enable_temporal_difference_filter = not analyzer.enable_temporal_difference_filter
            analyzer.prev_temporal_diff_frame = None
            if not analyzer.enable_temporal_difference_filter:
                profile_windows["temporal_diff_ms"].clear()
        if key == ord("g"):
            temporal_diff_modes = ["positive", "negative", "both"]
            next_idx = (temporal_diff_modes.index(analyzer.temporal_diff_polarity) + 1) % len(temporal_diff_modes)
            analyzer.temporal_diff_polarity = temporal_diff_modes[next_idx]
            analyzer.prev_temporal_diff_frame = None
        if key == ord("e"):
            analyzer.enable_screen_blend_equalization = not analyzer.enable_screen_blend_equalization
        if key == ord("b"):
            analyzer.enable_preprocess_blur = not analyzer.enable_preprocess_blur
        if key == ord("k"):
            editing_mask = True
            corner_points = []
            pending_corners = ["UL", "UR", "LR", "LL"]
            print("Mask edit mode: click UL, UR, LR, LL")
        if key == ord("l"):
            loaded = load_mask(mask_path)
            if loaded:
                analyzer.set_mask_points(loaded)
                print(f"Mask loaded: {mask_path}")
            else:
                print(f"Mask file not found: {mask_path}")
        if key == ord("L"):
            ok = save_mask(mask_path, analyzer.mask_points)
            print(f"Mask saved: {mask_path}" if ok else "Mask save skipped (no 4-point mask set)")
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            quality_idx = int(chr(key)) - 1
            analyzer.set_quality(**quality_presets[quality_idx])
            apply_flow_quality_for_current_mode(quality_idx)
        if step:
            freeze_key = cv2.waitKey(-1) & 0xFF
            if freeze_key == ord("q"):
                break
            if freeze_key != ord("n"):
                step = False

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()