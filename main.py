import ctypes
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from osc_sender import send_fused_wave_data
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
    line_colors=None,
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
    for idx, (line, (line_w, _line_h)) in enumerate(zip(lines, text_sizes)):
        if right_align:
            x = x0 + block_width - pad - line_w
        else:
            x = x0 + pad
        line_color = color if line_colors is None else line_colors[idx]
        cv2.putText(frame, line, (x, y), font, font_scale, line_color, thickness, cv2.LINE_AA)
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
    signal_smoothed,
    perf_stats,
    switches,
    profile_stats,
    show_cpu_profile,
):
    hud = frame.copy()
    color = (235, 235, 235)
    pace_color = (128, 128, 128)

    smooth = signal_smoothed
    signal_lines = [
        f"freq {smooth['wave_frequency_hz']:.2f} Hz ctr {smooth['freq_centroid_hz']:.2f} Hz",
        f"bump {smooth['bump_size_common'] * 1000.0:.2f}% spread {smooth['bump_size_spread'] * 1000.0:.2f}% max {smooth['bump_size_max'] * 1000.0:.2f}%",
        f"bump ctr(size) {smooth['bump_size_centroid'] * 100.0:.2f}% shape {smooth['bump_shape_roundness']:.2f}",
        f"dir {smooth['movement_direction_deg']:.1f} deg spd {smooth['movement_speed_norm']:.2f}",
        f"act {smooth['activity']:.2f} conf {smooth['confidence']:.2f}",
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
        f"quality: {quality_idx + 1}",
        f"flow: every {switches['flow_interval']} frame(s)",
    ]
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
        f"pace     {profile_stats.get('wait_ms', 0.0):.2f} | {profile_stats.get('wait_pct', 0.0):.1f}%",
        f"loop     {profile_stats.get('loop_ms', 0.0):.2f} | 100.0%",
    ]
    profile_line_colors = [color] * len(profile_lines)
    profile_line_colors[-2] = pace_color
    profile_w, profile_h = measure_text_block(profile_lines, font_scale=0.42, thickness=1, pad=8, line_gap=6)

    legend_lines = [
        "Keys",
        f"[D] mode: {mode_name}",
        f"[T] temporal mode: {switches['temporal_mode_label']}",
        f"[H] temporal diff: {'on' if switches['temporal_diff_filter'] else 'off'}",
        f"[G] tdiff polarity: {switches['temporal_diff_polarity']}",
        f"[E] screen: {switches['screen_blend_label']}",
        f"[B] blur: {switches['pre_blur_label']}",
        f"[A] gain: {switches['gain_label']}",
        f"[R] threshold overlay: {'on' if switches['threshold_overlay'] else 'off'}",
        f"[C] contour overlay: {'on' if switches['contour_overlay'] else 'off'}",
        f"[X] blob contours: {'on' if switches['contour_motion_filter'] else 'off'}",
        f"[U] line min len: {switches['contour_motion_threshold_label']}",
        f"[Z] straight lines: {'on' if switches['static_contours'] else 'off'}",
        f"[S] spectrum overlay: {'on' if switches['spectrum_overlay'] else 'off'}",
        f"[Y] cpu profile: {'on' if show_cpu_profile else 'off'}",
        f"[V] flow overlay: {'on' if switches['flow_overlay'] else 'off'}",
        f"[F] flow detail: {switches['flow_detail_mode']}",
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

    perf_top_y = hud.shape[0] - perf_h - 10
    if show_cpu_profile:
        profile_x = max(10, (hud.shape[1] - profile_w) // 2)
        draw_text_block(
            hud,
            profile_lines,
            profile_x,
            10,
            anchor="top_left",
            font_scale=0.42,
            thickness=1,
            color=color,
            line_colors=profile_line_colors,
            right_align=False,
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
        diameter = radius * 2
        # Offset each quadrant marker diagonally toward its image corner by 60% of one circle diameter.
        offset = int(round(diameter * 0.6))
        corner_dx = -offset if center[0] < (w * 0.5) else offset
        corner_dy = -offset if center[1] < (h * 0.5) else offset
        c_x = int(np.clip(center[0] + corner_dx, radius + 2, w - radius - 2))
        c_y = int(np.clip(center[1] + corner_dy, radius + 2, h - radius - 2))
        shifted_center = (c_x, c_y)

        cv2.circle(out, shifted_center, radius, (220, 220, 220), 1)

        if has_motion:
            arrow_len = int(np.clip(16 + strength * 8.0, 16, 44))
            dx = int(np.cos(np.radians(direction_deg)) * arrow_len)
            dy = int(np.sin(np.radians(direction_deg)) * arrow_len)
            cv2.arrowedLine(out, shifted_center, (shifted_center[0] + dx, shifted_center[1] + dy), (80, 190, 255), 2, tipLength=0.22)

        cv2.putText(out, label, (shifted_center[0] - 12, shifted_center[1] - radius - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)

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


def draw_spectral_slits_overlay(frame, slit_data, roi_gray=None):
    slit_rows = slit_data.get("slit_rows", []) if isinstance(slit_data, dict) else list(slit_data)
    slit_cols = slit_data.get("slit_cols", []) if isinstance(slit_data, dict) else []
    horizontal_spectra = slit_data.get("horizontal_spectra", []) if isinstance(slit_data, dict) else []
    vertical_spectra = slit_data.get("vertical_spectra", []) if isinstance(slit_data, dict) else []
    if not slit_rows and not slit_cols:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]
    WAVE_H = 20   # waveform height/width in pixels
    CIRC_R = 15   # quadrant center circle radius
    gray_ok = (
        roi_gray is not None
        and roi_gray.ndim == 2
        and roi_gray.shape == (h, w)
    )

    horizontal_samples = {int(item.get("row", -1)): item.get("samples") for item in horizontal_spectra if isinstance(item, dict)}
    vertical_samples = {int(item.get("col", -1)): item.get("samples") for item in vertical_spectra if isinstance(item, dict)}

    # Horizontal slits: amber line + green waveform drawn above the line
    for row in slit_rows:
        y = int(np.clip(row, 0, h - 1))
        cv2.line(out, (0, y), (w - 1, y), (255, 210, 60), 1)
        samples = horizontal_samples.get(y)
        if samples is not None and len(samples) == w:
            pixels = samples.astype(np.float32)
        elif gray_ok:
            pixels = roi_gray[y, :].astype(np.float32)
        else:
            pixels = None
        if pixels is not None:
            pmin, pmax = float(pixels.min()), float(pixels.max())
            norm = (pixels - pmin) / (pmax - pmin + 1e-6)
            xs = np.arange(w, dtype=np.int32)
            ys = np.clip((y - 1 - norm * WAVE_H).astype(np.int32), 0, h - 1)
            pts = np.column_stack([xs, ys])
            cv2.polylines(out, [pts], False, (60, 210, 120), 1)

    # Vertical slits: amber line + green waveform drawn to the left of the line
    for col in slit_cols:
        x = int(np.clip(col, 0, w - 1))
        cv2.line(out, (x, 0), (x, h - 1), (255, 210, 60), 1)
        samples = vertical_samples.get(x)
        if samples is not None and len(samples) == h:
            pixels = samples.astype(np.float32)
        elif gray_ok:
            pixels = roi_gray[:, x].astype(np.float32)
        else:
            pixels = None
        if pixels is not None:
            pmin, pmax = float(pixels.min()), float(pixels.max())
            norm = (pixels - pmin) / (pmax - pmin + 1e-6)
            ys = np.arange(h, dtype=np.int32)
            xs = np.clip((x - 1 - norm * WAVE_H).astype(np.int32), 0, w - 1)
            pts = np.column_stack([xs, ys])
            cv2.polylines(out, [pts], False, (60, 210, 120), 1)

    # Quadrant center circles: filled with pixel brightness, outlined in white
    if slit_rows and slit_cols:
        for ry in slit_rows:
            for cx in slit_cols:
                ry_c = int(np.clip(ry, 0, h - 1))
                cx_c = int(np.clip(cx, 0, w - 1))
                b = int(roi_gray[ry_c, cx_c]) if gray_ok else 128
                cv2.circle(out, (cx_c, ry_c), CIRC_R, (b, b, b), -1)
                cv2.circle(out, (cx_c, ry_c), CIRC_R, (220, 220, 220), 1)

    return out


def _draw_mini_spectrum(frame, xf, yf, x0, y0, plot_w=84, plot_h=54, max_freq=5.0, unit_label="", show_labels=True):
    h, w = frame.shape[:2]
    x0 = int(np.clip(x0, 0, max(0, w - plot_w)))
    y0 = int(np.clip(y0, 0, max(0, h - plot_h)))
    x1 = x0 + plot_w
    y1 = y0 + plot_h
    label_scale = 0.5
    (_, label_h), label_bl = cv2.getTextSize("0.0", cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1)
    label_strip_h = (label_h + label_bl + 4) if show_labels else 0
    bg_y0 = max(0, y0 - label_strip_h)

    draw_transparent_rect(frame, x0, bg_y0, plot_w, plot_h + label_strip_h, alpha=0.6)

    if xf is None or yf is None or len(xf) == 0 or len(yf) == 0:
        return

    left = x0 + 3
    right = x1 - 4
    top = y0 + 3
    bottom = y1 - 4
    if right <= left or bottom <= top:
        return

    mask = (xf >= 0.0) & (xf <= max_freq)
    if not np.any(mask):
        return
    xf_plot = xf[mask]
    yf_plot = yf[mask].astype(np.float32)
    max_amp = float(np.max(yf_plot))
    if max_amp <= 1e-9:
        return

    points = []
    for fx, amp in zip(xf_plot, yf_plot):
        px = left + int((float(fx) / max_freq) * (right - left))
        py = bottom - int((float(amp) / max_amp) * (bottom - top))
        points.append((int(np.clip(px, left, right)), int(np.clip(py, top, bottom))))
    if len(points) >= 2:
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], False, (210, 240, 120), 1)

    if show_labels:
        # Frequency range labels: start / middle / end of displayed band.
        f0 = 0.0
        f1 = max_freq * 0.5
        f2 = max_freq
        l0 = f"{f0:.1f}"
        l1 = f"{f1:.1f}"
        l2 = f"{f2:.1f}"

        label_y = bg_y0 + label_h + 1
        cv2.putText(frame, l0, (x0 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (200, 200, 200), 1, cv2.LINE_AA)
        (l1_w, _), _ = cv2.getTextSize(l1, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1)
        cv2.putText(frame, l1, (x0 + (plot_w - l1_w) // 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (200, 200, 200), 1, cv2.LINE_AA)
        (l2_w, _), _ = cv2.getTextSize(l2, cv2.FONT_HERSHEY_SIMPLEX, label_scale, 1)
        cv2.putText(frame, l2, (x1 - l2_w - 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, label_scale, (200, 200, 200), 1, cv2.LINE_AA)

    if unit_label:
        (u_w, u_h), _ = cv2.getTextSize(unit_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        unit_x = x0 + (plot_w - u_w) // 2
        unit_y = y0 + max(12, u_h + 4)
        cv2.putText(frame, unit_label, (unit_x, unit_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)


def draw_local_spectral_plots(frame, slit_data):
    out = frame.copy()
    h, w = out.shape[:2]
    slit_rows = slit_data.get("slit_rows", [])
    slit_cols = slit_data.get("slit_cols", [])
    plot_w = int(round(88 * 1.4))
    plot_h = 54

    # Spatial spectra for horizontal slits: top of slit, far left.
    h_specs = sorted(slit_data.get("horizontal_spectra", []), key=lambda s: s.get("row", 0))
    for idx, spec in enumerate(h_specs):
        row = int(np.clip(spec.get("row", 0), 0, h - 1))
        # Show frequency labels only on upper-left spatial display.
        _draw_mini_spectrum(
            out,
            spec.get("xf", np.array([])),
            spec.get("yf", np.array([])),
            5,
            row - (plot_h + 4),
            plot_w=plot_w,
            plot_h=plot_h,
            max_freq=20.0,
            unit_label="c/f",
            show_labels=(idx == 0),
        )

    # Spatial spectra for vertical slits: bottom, beside each slit.
    v_specs = slit_data.get("vertical_spectra", [])
    if len(v_specs) >= 1:
        left_col = int(np.clip(v_specs[0].get("col", slit_cols[0] if slit_cols else 0), 0, w - 1))
        _draw_mini_spectrum(out, v_specs[0].get("xf", np.array([])), v_specs[0].get("yf", np.array([])), left_col + 5, h - (plot_h + 4), plot_w=plot_w, plot_h=plot_h, max_freq=20.0, unit_label="c/f", show_labels=False)
    if len(v_specs) >= 2:
        right_col = int(np.clip(v_specs[1].get("col", slit_cols[1] if len(slit_cols) > 1 else w - 1), 0, w - 1))
        _draw_mini_spectrum(out, v_specs[1].get("xf", np.array([])), v_specs[1].get("yf", np.array([])), right_col - (plot_w + 5), h - (plot_h + 4), plot_w=plot_w, plot_h=plot_h, max_freq=20.0, unit_label="c/f", show_labels=False)

    # Temporal spectra near quadrant points, nudged toward image center with 5 px gap.
    qp = slit_data.get("quadrant_points", {})
    qt = slit_data.get("quadrant_temporal", {})
    center_x = w * 0.5
    center_y = h * 0.5
    for label, point in qp.items():
        if label not in qt:
            continue
        cx = int(np.clip(point.get("x", 0), 0, w - 1))
        cy = int(np.clip(point.get("y", 0), 0, h - 1))
        toward_center_x = 1 if cx < center_x else -1
        toward_center_y = 1 if cy < center_y else -1

        if toward_center_x > 0 and toward_center_y > 0:  # TL
            px = cx + 5
            py = cy + 5
        elif toward_center_x < 0 and toward_center_y > 0:  # TR
            px = cx - 5 - plot_w
            py = cy + 5
        elif toward_center_x > 0 and toward_center_y < 0:  # BL
            px = cx + 5
            py = cy - 5 - plot_h
        else:  # BR
            px = cx - 5 - plot_w
            py = cy - 5 - plot_h

        # Show frequency labels only on lower-right temporal display.
        _draw_mini_spectrum(out, qt[label].get("xf", np.array([])), qt[label].get("yf", np.array([])), px, py, plot_w=plot_w, plot_h=plot_h, max_freq=5.0, unit_label="Hz", show_labels=(label == "BR"))

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
        # X toggles visibility of blob-like contours (green)
        # Z toggles visibility of straight-line contours (yellow)
        # U adjusts straight-line minimum length threshold.

        # Show blob-like contours in green if X is enabled
        if analyzer.enable_contour_motion_filter:
            blob_contours = analysis["contours"]
            if blob_contours:
                cv2.drawContours(out, blob_contours, -1, (30, 250, 70), 2)
                # Highlight the largest blob-like contour in red
                largest_idx = int(np.argmax([cv2.contourArea(c) for c in blob_contours]))
                cv2.drawContours(out, blob_contours, largest_idx, (0, 0, 255), 2)

        # Show straight-line contours in yellow if Z is enabled
        if analyzer.show_static_contours:
            straight_line_contours = analysis.get("static_contours", [])
            if straight_line_contours:
                cv2.drawContours(out, straight_line_contours, -1, (0, 220, 255), 2)

    if flow_overlay:
        out = draw_flow_overlay(out, analyzer.last_flow)
        out = draw_quadrant_flow_arrows(out, analyzer.last_flow)

    if spectrum_overlay:
        slit = analysis["slit_data"]
        out = draw_spectral_slits_overlay(out, slit, analysis["preprocess_display"])
        out = draw_local_spectral_plots(out, slit)

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
    profile_window_seconds = 2.0
    profile_update_hz = 2.0
    profile_update_interval_s = 1.0 / profile_update_hz
    status_panel_update_hz = 4.0
    status_panel_update_interval_s = 1.0 / status_panel_update_hz
    profile_window_len = max(4, int(round(fps * profile_window_seconds)))
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
    show_cpu_profile = True
    temporal_modes = [
        {"label": "off", "enabled": False, "seconds": None, "output": "change"},
        {"label": "lp 0.5s", "enabled": True, "seconds": 0.5, "output": "lowpass"},
        {"label": "lp 2s", "enabled": True, "seconds": 2.0, "output": "lowpass"},
        {"label": "chg 0.5s", "enabled": True, "seconds": 0.5, "output": "change"},
        {"label": "chg 2s", "enabled": True, "seconds": 2.0, "output": "change"},
    ]
    temporal_mode_idx = 4

    def apply_temporal_mode(mode_idx):
        mode = temporal_modes[mode_idx]
        analyzer.enable_temporal_change_filter = mode["enabled"]
        if mode["seconds"] is not None:
            analyzer.temporal_filter_seconds = float(mode["seconds"])
        analyzer.temporal_filter_output_mode = str(mode.get("output", "change"))
        analyzer.prev_temporal_float = None
        analyzer.prev_temporal_u8 = None
        analyzer.temporal_change_u8 = None
        if not analyzer.enable_temporal_change_filter:
            profile_windows["temporal_filter_ms"].clear()

    editing_mask = False
    pending_corners = ["UL", "UR", "LR", "LL"]
    corner_points = []
    last_analysis = None
    frame_timestamps = deque(maxlen=profile_window_len)
    processing_fps_values = deque(maxlen=profile_window_len)
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
    profile_windows = {k: deque(maxlen=profile_window_len) for k in profile_keys}
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
    }
    next_profile_update_t = 0.0
    current_perf_stats = {
        "source_fps": float(fps),
        "effective_fps": 0.0,
        "processing_fps": 0.0,
        "playback_ratio_pct": 0.0,
    }
    current_switches = None
    current_quality_idx = quality_idx
    current_signal_smoothed = {
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
    next_status_panel_update_t = 0.0
    # Retained variable names for compatibility; values now represent minimum straight-line length (px).
    contour_motion_threshold_presets = [20.0, 30.0, 45.0, 60.0]
    contour_motion_threshold_idx = 0
    if contour_motion_threshold_presets:
        diffs = [abs(float(analyzer.contour_motion_threshold_px) - p) for p in contour_motion_threshold_presets]
        contour_motion_threshold_idx = int(np.argmin(diffs))
        analyzer.contour_motion_threshold_px = contour_motion_threshold_presets[contour_motion_threshold_idx]

    apply_temporal_mode(temporal_mode_idx)

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
    frame_period_s = (1.0 / float(fps)) if fps > 0 else 0.0
    next_frame_deadline = time.perf_counter()

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
            "flow_overlay": show_flow_overlay,
            "contour_overlay": show_contour_overlay,
            "threshold_overlay": show_threshold_overlay,
            "spectrum_overlay": show_spectrum_overlay,
            "temporal_mode_label": temporal_modes[temporal_mode_idx]["label"],
            "temporal_filter": analyzer.enable_temporal_change_filter,
            "screen_blend_label": ["off", "1x", "2x"][max(0, min(2, analyzer.screen_blend_mode))],
            "pre_blur_label": ["off", "small", "large"][max(0, min(2, analyzer.blur_mode))],
            "gain_label": ["off", "-25%", "+50%", "auto"][max(0, min(3, analyzer.gain_mode))],
            "contour_motion_filter": analyzer.enable_contour_motion_filter,
            "static_contours": analyzer.show_static_contours,
            "contour_motion_threshold_label": f"{float(analyzer.contour_motion_threshold_px):.1f}px",
        }

        if cap_t1 >= next_status_panel_update_t or current_switches is None:
            current_perf_stats = perf_stats.copy()
            current_switches = switches.copy()
            current_quality_idx = quality_idx
            current_signal_smoothed = dict(analysis["smoothed"])
            next_status_panel_update_t = cap_t1 + status_panel_update_interval_s

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
        )
        display = draw_status_hud(
            display,
            mode_name,
            current_quality_idx,
            editing_mask,
            pending_corners,
            current_signal_smoothed,
            current_perf_stats,
            current_switches,
            current_profile_stats,
            show_cpu_profile,
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

        wait_t0 = time.perf_counter()
        if frame_period_s > 0.0:
            next_frame_deadline += frame_period_s
            remaining_s = next_frame_deadline - wait_t0
            if remaining_s > 0.0:
                time.sleep(remaining_s)
            else:
                next_frame_deadline = wait_t0
        key = cv2.pollKey() & 0xFF
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

        if loop_end >= next_profile_update_t:
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

            current_profile_stats = {
                **avg_profile_ms,
                **avg_profile_pct,
            }
            next_profile_update_t = loop_end + profile_update_interval_s

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
        if key == ord("x"):
            analyzer.enable_contour_motion_filter = not analyzer.enable_contour_motion_filter
        if key == ord("u"):
            if contour_motion_threshold_presets:
                contour_motion_threshold_idx = (contour_motion_threshold_idx + 1) % len(contour_motion_threshold_presets)
                analyzer.contour_motion_threshold_px = contour_motion_threshold_presets[contour_motion_threshold_idx]
        if key == ord("z"):
            analyzer.show_static_contours = not analyzer.show_static_contours
        if key == ord("r"):
            show_threshold_overlay = not show_threshold_overlay
        if key == ord("s"):
            show_spectrum_overlay = not show_spectrum_overlay
        if key == ord("y"):
            show_cpu_profile = not show_cpu_profile
        if key == ord("m"):
            analyzer.show_mask = not analyzer.show_mask
        if key == ord("f"):
            next_idx = (flow_detail_modes.index(flow_detail_mode) + 1) % len(flow_detail_modes)
            flow_detail_mode = flow_detail_modes[next_idx]
            apply_flow_quality_for_current_mode(quality_idx)
        if key == ord("t"):
            temporal_mode_idx = (temporal_mode_idx + 1) % len(temporal_modes)
            apply_temporal_mode(temporal_mode_idx)
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
            analyzer.screen_blend_mode = (int(analyzer.screen_blend_mode) + 1) % 3
            profile_windows["screen_blend_ms"].clear()
        if key == ord("b"):
            analyzer.blur_mode = (int(analyzer.blur_mode) + 1) % 3
            profile_windows["pre_blur_ms"].clear()
        if key == ord("a"):
            analyzer.gain_mode = (int(analyzer.gain_mode) + 1) % 4
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


