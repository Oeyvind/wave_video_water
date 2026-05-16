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
    roi = frame[y:y2, x:x2]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x2 - x - 1, y2 - y - 1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)


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
    flow_data=None,
    pyramid_data=None,
    wavelength_data=None,
    lbp_data=None,
):
    hud = frame.copy()
    color = (235, 235, 235)
    pace_color = (128, 128, 128)

    smooth = signal_smoothed
    signal_lines = [
        f"bump {smooth['bump_size_common'] * 1000.0:.2f}% spread {smooth['bump_size_spread'] * 1000.0:.2f}% max {smooth['bump_size_max'] * 1000.0:.2f}%",
        f"bump ctr(size) {smooth['bump_size_centroid'] * 100.0:.2f}% shape {smooth['bump_shape_roundness']:.2f}",
        f"act {smooth['activity']:.2f} conf {smooth['confidence']:.2f}",
    ]
    fd = flow_data or {}
    lbp = lbp_data or {}
    signal_lines += [
        f"flow fast: act {fd.get('fast_activity', 0.0):.2f}  spd {fd.get('fast_speed_norm', 0.0):.2f}  dir {fd.get('fast_direction_deg', 0.0):.0f}  coh {fd.get('fast_coherence', 0.0):.2f}",
        f"flow slow: act {fd.get('slow_activity', 0.0):.2f}  spd {fd.get('slow_speed_norm', 0.0):.2f}  dir {fd.get('slow_direction_deg', 0.0):.0f}  coh {fd.get('slow_coherence', 0.0):.2f}",
        f"flow dir tgt {fd.get('directional_target_deg', 225.0):.0f}deg ({fd.get('directional_source', 'fast')})  support {fd.get('directional_global', {}).get('support', 0.0):.2f}  hit {fd.get('directional_global', {}).get('hit_ratio', 0.0):.2f}",
        f"flow best: quad {fd.get('directional_best_quadrant', '-')} {fd.get('directional_best_quadrant_support', 0.0):.2f}  stripe {fd.get('directional_best_stripe', '-')} {fd.get('directional_best_stripe_support', 0.0):.2f}",
        f"lbp: rough {lbp.get('lbp_roughness', 0.0):.2f}  uniform {lbp.get('lbp_uniform_ratio', 0.0):.2f}  entr {lbp.get('lbp_entropy', 0.0):.2f}",
    ]

    _, signal_h = measure_text_block(signal_lines, font_scale=0.52, thickness=1, pad=8, line_gap=6)
    signal_y = max(10, (hud.shape[0] - signal_h) // 2)

    draw_text_block(
        hud,
        signal_lines,
        10,
        signal_y,
        anchor="top_left",
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
        f"flow fast: every {switches['flow_interval']} frame(s) ({'auto' if switches['flow_update_auto'] else 'manual'})",
        f"flow slow: every {switches['flow_slow_interval']} frame(s)",
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
        f"pyramid  {profile_stats.get('pyramid_ms', 0.0):.2f} | {profile_stats.get('pyramid_pct', 0.0):.1f}%",
        f"lbp     {profile_stats.get('lbp_ms', 0.0):.2f} | {profile_stats.get('lbp_pct', 0.0):.1f}%",
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
        f"[A] gain: {switches['gain_label']}",
        f"[B] blur: {switches['pre_blur_label']}",
        f"[R] threshold: {'on' if switches['threshold_overlay'] else 'off'}",
        f"[C] contour overlay: {'on' if switches['contour_overlay'] else 'off'}",
        f"[U] line min len: {switches['contour_motion_threshold_label']}",
        f"[S] pyramid overlay: {'on' if switches['spectrum_overlay'] else 'off'}",
        f"[W] texture overlay: {'on' if switches['texture_overlay'] else 'off'}",
        f"[Y] cpu profile: {'on' if show_cpu_profile else 'off'}",
        f"[V] flow overlay: {'on' if switches['flow_overlay'] else 'off'}",
        f"[J] LBP analysis: {'on' if switches['lbp_analysis'] else 'off'}",
        f"[F] flow detail: {switches['flow_detail_mode']}",
        f"[I] fast interval auto: {'on' if switches['flow_update_auto'] else 'off'}",
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


def draw_flow_overlay(frame, flow, color=(30, 180, 255), disp_scale=2.0):
    if flow is None:
        return frame
    out = frame.copy()
    h, w = out.shape[:2]
    fh, fw = flow.shape[:2]
    step = 16
    sx = w / float(fw)
    sy = h / float(fh)
    line_thickness = 3

    for y in range(0, fh, step):
        for x in range(0, fw, step):
            dx, dy = flow[y, x]
            p0 = (int(x * sx), int(y * sy))
            p1 = (int((x + dx * disp_scale) * sx), int((y + dy * disp_scale) * sy))
            cv2.line(out, p0, p1, color, line_thickness)
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


def _flow_arrow_centers(frame_h, frame_w):
    q_radius = 34
    diameter = q_radius * 2
    offset = int(round(diameter * 0.6))

    centers = {}
    for label, base_center in (
        ("UL", (int(frame_w * 0.25), int(frame_h * 0.25))),
        ("UR", (int(frame_w * 0.75), int(frame_h * 0.25))),
        ("LL", (int(frame_w * 0.25), int(frame_h * 0.75))),
        ("LR", (int(frame_w * 0.75), int(frame_h * 0.75))),
    ):
        corner_dx = -offset if base_center[0] < (frame_w * 0.5) else offset
        corner_dy = -offset if base_center[1] < (frame_h * 0.5) else offset
        c_x = int(np.clip(base_center[0] + corner_dx, q_radius + 2, frame_w - q_radius - 2))
        c_y = int(np.clip(base_center[1] + corner_dy, q_radius + 2, frame_h - q_radius - 2))
        centers[label] = (c_x, c_y)

    centers["G"] = (int(frame_w * 0.5), int(frame_h * 0.5))
    return centers


def _draw_vertical_slider(frame, x0, y0, value, label, slider_h=68, slider_w=10, color=(80, 170, 255), value_max=300.0):
    x0 = int(x0)
    y0 = int(y0)
    slider_h = int(max(24, slider_h))
    slider_w = int(max(6, slider_w))
    value = 0.0 if value is None else float(max(0.0, value))
    norm = float(np.clip(value / max(value_max, 1e-6), 0.0, 1.0))

    draw_transparent_rect(frame, x0 - 4, y0 - 16, slider_w + 8, slider_h + 22, alpha=0.52)
    base_y = y0 + slider_h
    fill_h = int(max(1, round(norm * slider_h))) if value > 0.0 else 0
    cv2.rectangle(frame, (x0, y0), (x0 + slider_w, base_y), (140, 140, 140), 1)
    if fill_h > 0:
        cv2.rectangle(frame, (x0, base_y - fill_h), (x0 + slider_w, base_y), color, -1)
    cv2.putText(frame, label, (x0 - 2, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (225, 225, 225), 1, cv2.LINE_AA)


def draw_quadrant_flow_arrows(frame, flow, arrow_color=(80, 190, 255), global_color=(120, 220, 255),
                              circle_color=(220, 220, 220), global_rate_hz=None, rate_side=None):
    if flow is None:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]
    fh, fw = flow.shape[:2]
    half_h = fh // 2
    half_w = fw // 2
    centers = _flow_arrow_centers(h, w)

    quadrants = [
        ("UL", flow[0:half_h, 0:half_w], (int(w * 0.25), int(h * 0.25))),
        ("UR", flow[0:half_h, half_w:fw], (int(w * 0.75), int(h * 0.25))),
        ("LL", flow[half_h:fh, 0:half_w], (int(w * 0.25), int(h * 0.75))),
        ("LR", flow[half_h:fh, half_w:fw], (int(w * 0.75), int(h * 0.75))),
    ]

    for label, block, center in quadrants:
        direction_deg, strength, has_motion = _quadrant_vector_stats(block)
        radius = 34
        shifted_center = centers[label]

        cv2.circle(out, shifted_center, radius, circle_color, 1)

        if has_motion:
            q_color = arrow_color
            arrow_len = int(np.clip(16 + strength * 8.0, 16, 44))
            dx = int(np.cos(np.radians(direction_deg)) * arrow_len)
            dy = int(np.sin(np.radians(direction_deg)) * arrow_len)
            cv2.arrowedLine(out, shifted_center, (shifted_center[0] + dx, shifted_center[1] + dy), q_color, 2, tipLength=0.22)

        cv2.putText(out, label, (shifted_center[0] - 12, shifted_center[1] - radius - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)

    # Global summary arrow
    g_direction_deg, g_strength, g_has_motion = _quadrant_vector_stats(flow)
    g_center = centers["G"]
    g_radius = 42
    cv2.circle(out, g_center, g_radius, circle_color, 1)
    if g_has_motion:
        g_color = global_color
        g_len = int(np.clip(20 + g_strength * 10.0, 20, 56))
        g_dx = int(np.cos(np.radians(g_direction_deg)) * g_len)
        g_dy = int(np.sin(np.radians(g_direction_deg)) * g_len)
        cv2.arrowedLine(out, g_center, (g_center[0] + g_dx, g_center[1] + g_dy), g_color, 3, tipLength=0.22)

    if global_rate_hz is not None and rate_side in ("left", "right"):
        rate_text = f"{float(global_rate_hz):.1f}Hz"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(rate_text, font, font_scale, thickness)
        # Anchor near the upper-left / upper-right arc (about 45 degrees),
        # then offset by 4 px outward so text visually hugs the circle.
        arc_dx = int(round(g_radius * 0.72))
        arc_dy = int(round(g_radius * 0.72))
        y = g_center[1] - arc_dy + (th // 2)
        if rate_side == "left":
            # Right-align left label to a point 4 px left of the UL arc.
            x = (g_center[0] - arc_dx - 4) - tw
        else:
            # Left-align right label to a point 4 px right of the UR arc.
            x = g_center[0] + arc_dx + 4
        cv2.putText(out, rate_text, (x, y), font, font_scale, global_color, thickness, cv2.LINE_AA)

    cv2.putText(out, "G", (g_center[0] - 7, g_center[1] - g_radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)

    return out


def draw_lbp_overlay(frame, lbp_data):
    """LBP overlay with aggregate gauges and compact grouped histogram."""
    lbp = lbp_data or {}
    if not lbp:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]
    scale = 0.8
    panel_w = int(round(390 * scale))
    panel_h = int(round(186 * scale))
    # Center-bottom placement, 8 px lower than the former lower-left baseline.
    x0 = (w - panel_w) // 2
    y0 = h - panel_h - 8
    draw_transparent_rect(out, x0, y0, panel_w, panel_h, alpha=0.58)
    cv2.putText(out, "LBP histogram (16 bins)", (x0 + 6, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1, cv2.LINE_AA)

    # Aggregate bars
    roughness = float(lbp.get("lbp_roughness", 0.0))
    entropy = float(lbp.get("lbp_entropy", 0.0))
    uniform = float(lbp.get("lbp_uniform_ratio", 0.0))
    agg_x = x0 + 8
    agg_y = y0 + 24
    agg_w = panel_w - 16
    # 60% of previous horizontal bar thickness (10 -> 6)
    agg_h = 6
    agg_gap = 12
    for label, val, col in [
        ("roughness", roughness, (200, 120, 50)),
        ("entropy", min(entropy / 8.0, 1.0), (170, 90, 200)),
        ("uniform", uniform, (60, 170, 210)),
    ]:
        filled = max(0, int(round(val * agg_w)))
        cv2.rectangle(out, (agg_x, agg_y), (agg_x + agg_w, agg_y + agg_h), (72, 72, 72), 1)
        if filled > 0:
            cv2.rectangle(out, (agg_x, agg_y), (agg_x + filled, agg_y + agg_h), col, -1)
        cv2.putText(out, f"{label} {val:.2f}", (agg_x, agg_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (208, 208, 208), 1, cv2.LINE_AA)
        agg_y += agg_h + agg_gap

    # Bottom histogram with 16 grouped bins.
    hist = lbp.get("lbp_histogram_16", [0.0] * 16)
    hist_vals = [float(v) for v in hist]
    bin_count = max(1, len(hist_vals))
    hist_max = max(max(hist_vals), 1e-9)

    hist_x0 = x0 + 8
    hist_y0 = y0 + 82
    hist_h = 54
    hist_gap = 2
    bin_w = max(2, int((panel_w - 16 - hist_gap * max(0, bin_count - 1)) / bin_count))
    cv2.putText(out, "grouped codes", (hist_x0, hist_y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (185, 185, 185), 1, cv2.LINE_AA)

    for idx, val in enumerate(hist_vals):
        x = hist_x0 + idx * (bin_w + hist_gap)
        hpx = int(round((val / hist_max) * hist_h))
        cv2.rectangle(out, (x, hist_y0), (x + bin_w, hist_y0 + hist_h), (70, 70, 70), 1)
        if hpx > 0:
            cv2.rectangle(out, (x, hist_y0 + hist_h - hpx), (x + bin_w, hist_y0 + hist_h), (150, 150, 210), -1)

    # Label key positions so bin meaning is clear.
    label_y = hist_y0 + hist_h + 12
    b1_x = hist_x0
    b9_x = hist_x0 + 7 * (bin_w + hist_gap)
    b16_x = hist_x0 + 15 * (bin_w + hist_gap)
    cv2.putText(out, "1", (b1_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (165, 165, 165), 1, cv2.LINE_AA)
    cv2.putText(out, "8", (b9_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (165, 165, 165), 1, cv2.LINE_AA)
    cv2.putText(out, "16", (b16_x - 6, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (165, 165, 165), 1, cv2.LINE_AA)

    return out


# Pyramid texture band colors: yellow-wide, coarse, extra-coarse.
# Ordered as yellow -> blue -> purple.
_PYRAMID_BAND_COLORS = [
    (0, 220, 255),
    (255, 140, 80),
    (220, 60, 220),
]


def _centroid_to_color(centroid, brightness=1.0):
    """Interpolate through pyramid band colors using a centroid in [0, 1].

    centroid=0 → yellow-wide, centroid=1 → extra-coarse (purple).
    brightness scales the result, e.g. 0.65 for dimmed slow-flow arrows.
    """
    colors = _PYRAMID_BAND_COLORS
    n = len(colors) - 1
    t = float(np.clip(centroid, 0.0, 1.0)) * n
    lo = int(t)
    hi = min(lo + 1, n)
    frac = t - lo
    c0, c1 = colors[lo], colors[hi]
    b = float(np.clip(brightness, 0.0, 1.0))
    return (
        int((c0[0] + frac * (c1[0] - c0[0])) * b),
        int((c0[1] + frac * (c1[1] - c0[1])) * b),
        int((c0[2] + frac * (c1[2] - c0[2])) * b),
    )


def _rate_hz_to_color(rate_hz, min_hz=0.5, max_hz=30.0):
    """Map analysis update rate (Hz) to palette color using a log scale.

    Low rate (slow analysis) maps to low palette index; high rate (fast analysis)
    maps to high palette index.
    """
    lo = max(1e-6, float(min_hz))
    hi = max(lo + 1e-6, float(max_hz))
    rate = float(np.clip(rate_hz, lo, hi))
    norm = (np.log(rate) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return _centroid_to_color(norm)


def _draw_pyramid_bar_group(frame, x0, y0, bands, label, bar_w=8, bar_h=52, gap=4, temporal_bands=None, centroid=None):
    bands = list(bands) if bands is not None else [0.0] * len(_PYRAMID_BAND_COLORS)
    if len(bands) < len(_PYRAMID_BAND_COLORS):
        bands = bands + [0.0] * (len(_PYRAMID_BAND_COLORS) - len(bands))
    bar_count = len(_PYRAMID_BAND_COLORS)
    group_w = (bar_w * bar_count) + (gap * max(0, bar_count - 1))
    temporal_present = temporal_bands is not None
    centroid_present = centroid is not None
    temporal_h = bar_h          # same height as spatial bars
    centroid_h = 12
    t_centroid_rows = centroid_h + 12 if temporal_present else 0
    group_h = bar_h + 18 + (centroid_h + 12 if centroid_present else 0) + (temporal_h + 16 if temporal_present else 0) + t_centroid_rows

    draw_transparent_rect(frame, x0 - 4, y0 - 14, group_w + 8, group_h + 8, alpha=0.52)
    cv2.putText(frame, label, (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (225, 225, 225), 1, cv2.LINE_AA)

    base_y = y0 + bar_h
    for idx in range(bar_count):
        val = float(np.clip(bands[idx], 0.0, 1.0))
        hpx = int(max(1, round(val * bar_h)))
        bx0 = x0 + idx * (bar_w + gap)
        bx1 = bx0 + bar_w
        by0 = base_y - hpx
        cv2.rectangle(frame, (bx0, by0), (bx1, base_y), _PYRAMID_BAND_COLORS[idx], -1)
        cv2.rectangle(frame, (bx0, y0), (bx1, base_y), (140, 140, 140), 1)

    temporal_offset = 0
    if centroid_present:
        cval = float(np.clip(centroid, 0.0, 1.0))
        c_y0 = base_y + 6
        c_h = centroid_h
        c_mid = c_y0 + (c_h // 2)
        rail_x0 = x0
        rail_x1 = x0 + group_w
        cv2.line(frame, (rail_x0, c_mid), (rail_x1, c_mid), (170, 170, 170), 1, cv2.LINE_AA)
        for t in range(bar_count):
            tx = x0 + int(round((group_w - 1) * (t / max(1, bar_count - 1))))
            cv2.line(frame, (tx, c_mid - 2), (tx, c_mid + 2), (110, 110, 110), 1, cv2.LINE_AA)
        marker_x = x0 + int(round(cval * (group_w - 1)))
        cv2.rectangle(frame, (marker_x - 2, c_y0), (marker_x + 2, c_y0 + c_h), (240, 240, 240), -1)
        temporal_offset = c_h + 12

    if temporal_present:
        tvals = list(temporal_bands)
        if len(tvals) < bar_count:
            tvals = tvals + [0.0] * (bar_count - len(tvals))
        t_y0 = base_y + 12 + temporal_offset
        t_base_y = t_y0 + temporal_h
        cv2.putText(frame, "T", (x0 - 12, t_base_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (205, 205, 205), 1, cv2.LINE_AA)
        for idx in range(bar_count):
            val = float(np.clip(tvals[idx], 0.0, 1.0))
            hpx = int(max(1, round(val * temporal_h)))
            bx0 = x0 + idx * (bar_w + gap)
            bx1 = bx0 + bar_w
            by0 = t_base_y - hpx
            c = _PYRAMID_BAND_COLORS[idx]
            dim_c = (int(c[0] * 0.65), int(c[1] * 0.65), int(c[2] * 0.65))
            cv2.rectangle(frame, (bx0, by0), (bx1, t_base_y), dim_c, -1)
            cv2.rectangle(frame, (bx0, t_y0), (bx1, t_base_y), (120, 120, 120), 1)

        # Temporal centroid rail — weighted mean of temporal band indices
        t_arr = np.asarray(tvals[:bar_count], dtype=np.float32)
        t_total = float(np.sum(t_arr)) + 1e-9
        t_centroid = float(np.dot(np.arange(bar_count, dtype=np.float32), t_arr) / (t_total * (bar_count - 1)))
        tc_y0 = t_base_y + 6
        tc_h = centroid_h
        tc_mid = tc_y0 + (tc_h // 2)
        cv2.line(frame, (x0, tc_mid), (x0 + group_w, tc_mid), (130, 130, 130), 1, cv2.LINE_AA)
        for t in range(bar_count):
            tx = x0 + int(round((group_w - 1) * (t / max(1, bar_count - 1))))
            cv2.line(frame, (tx, tc_mid - 2), (tx, tc_mid + 2), (90, 90, 90), 1, cv2.LINE_AA)
        tc_marker_x = x0 + int(round(float(np.clip(t_centroid, 0.0, 1.0)) * (group_w - 1)))
        cv2.rectangle(frame, (tc_marker_x - 2, tc_y0), (tc_marker_x + 2, tc_y0 + tc_h), (180, 180, 180), -1)


def draw_pyramid_texture_bars(frame, pyramid_data, wavelength_data=None):
    if not isinstance(pyramid_data, dict):
        return frame

    out = frame
    h, w = out.shape[:2]
    g = pyramid_data.get("global_bands", [0.0, 0.0, 0.0])
    g_t = pyramid_data.get("temporal_band_activity", [0.0, 0.0, 0.0])
    g_ctr = float(pyramid_data.get("scale_centroid", 0.0))
    q = pyramid_data.get("quadrant_bands", {})
    q_t = pyramid_data.get("quadrant_temporal_bands", {})
    q_ctr = pyramid_data.get("quadrant_scale_centroids", {})
    wd = wavelength_data or {}
    q_wl = wd.get("quadrants", {})
    g_wl = wd.get("wavelength_px")

    centers = _flow_arrow_centers(h, w)

    bar_count = len(_PYRAMID_BAND_COLORS)
    group_w = (8 * bar_count) + (4 * max(0, bar_count - 1))
    # spatial(52) + label(18) + s_centroid(24) + T_bars(52) + T_gap(16) + t_centroid(24)
    group_h = 52 + 18 + 24 + 52 + 16 + 24
    group_h_global = group_h
    q_radius = 34
    g_radius = 42
    pad = 10

    for label in ("UL", "UR", "LL", "LR"):
        cx, cy = centers[label]
        if label in ("UL", "LL"):
            # Bars on inward side (toward horizontal center), wavelength on opposite side.
            x = cx + q_radius + pad
            wl_x = cx - q_radius - pad - 10
        else:
            x = cx - q_radius - pad - group_w
            wl_x = cx + q_radius + pad
        y = cy - (group_h // 2)
        x = int(np.clip(x, 14, max(14, w - 14 - group_w)))
        y = int(np.clip(y, 24, max(24, h - 24 - group_h)))
        _draw_pyramid_bar_group(out, x, y, q.get(label, [0.0] * bar_count), label,
                    temporal_bands=q_t.get(label), centroid=q_ctr.get(label))
        wl_val = (q_wl.get(label) or {}).get("wavelength_px")
        wl_y = int(np.clip(cy - 34, 24, max(24, h - 24 - 68)))
        wl_x = int(np.clip(wl_x, 10, max(10, w - 10 - 10)))
        _draw_vertical_slider(out, wl_x, wl_y, wl_val, label="WL", slider_h=68, slider_w=10, color=(30, 140, 255), value_max=300.0)

    g_cx, g_cy = centers["G"]
    g_bar_x = int(np.clip(g_cx + g_radius + pad, 14, max(14, w - 14 - group_w)))
    g_bar_y = int(np.clip(g_cy + 14, 24, max(24, h - 24 - group_h_global)))
    _draw_pyramid_bar_group(out, g_bar_x, g_bar_y, g, "GLOBAL S/T", temporal_bands=g_t, centroid=g_ctr)
    g_wl_x = int(np.clip(g_cx - g_radius - pad - 10, 10, max(10, w - 10 - 10)))
    g_wl_y = int(np.clip(g_cy + 14, 24, max(24, h - 24 - 68)))
    _draw_vertical_slider(out, g_wl_x, g_wl_y, g_wl, label="WL", slider_h=68, slider_w=10, color=(30, 140, 255), value_max=300.0)
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
        # Show pre-threshold filtered signal in filtered mode.
        # Threshold display remains controlled by the R overlay toggle.
        out = cv2.cvtColor(analysis["preprocess_display"], cv2.COLOR_GRAY2BGR)
    else:
        out = base_frame.copy()

    if threshold_overlay:
        thr_rgb = cv2.cvtColor(analysis["threshold"], cv2.COLOR_GRAY2BGR)
        # Threshold overlay now shows only analysis-fed threshold content.
        out = cv2.addWeighted(out, 0.2, thr_rgb, 0.8, 0)

    if contour_overlay:
        # Blob-like contours in green and straight-line contours in yellow.
        # U adjusts straight-line minimum length threshold.
        blob_contours = analysis["contours"]
        if blob_contours:
            cv2.drawContours(out, blob_contours, -1, (30, 250, 70), 2)
            # Highlight the largest blob-like contour in red
            largest_idx = int(np.argmax([cv2.contourArea(c) for c in blob_contours]))
            cv2.drawContours(out, blob_contours, largest_idx, (0, 0, 255), 2)

        straight_line_contours = analysis.get("static_contours", [])
        if straight_line_contours:
            cv2.drawContours(out, straight_line_contours, -1, (0, 220, 255), 2)

    if flow_overlay:
        fast_rate_hz = float(analyzer.fps) / max(1, int(analyzer.flow_update_interval))
        slow_rate_hz = float(analyzer.fps) / max(1, int(analyzer.flow_slow_interval))
        fast_arrow_color = _rate_hz_to_color(fast_rate_hz)
        slow_arrow_color = _rate_hz_to_color(slow_rate_hz)

        # Fast flow: color keyed by fast analysis rate
        out = draw_flow_overlay(out, analyzer.last_flow, color=(30, 180, 255), disp_scale=2.0)
        out = draw_quadrant_flow_arrows(out, analyzer.last_flow,
                                        arrow_color=fast_arrow_color, global_color=fast_arrow_color,
                                        circle_color=(200, 200, 200),
                                        global_rate_hz=fast_rate_hz, rate_side="right",
                                        )
        # Slow flow: color keyed by slow analysis rate
        slow_disp_scale = 2.0 / max(1, analyzer.flow_slow_interval)
        out = draw_flow_overlay(out, analyzer.last_flow_slow, color=(30, 100, 255), disp_scale=slow_disp_scale)
        out = draw_quadrant_flow_arrows(out, analyzer.last_flow_slow,
                                        arrow_color=slow_arrow_color, global_color=slow_arrow_color,
                                        circle_color=(160, 160, 160),
                                        global_rate_hz=slow_rate_hz, rate_side="left",
                                        )

    if spectrum_overlay:
        out = draw_pyramid_texture_bars(out, analysis.get("pyramid_data"), analysis.get("wavelength_data"))

    return out


def main():
    video_source, source_path = select_video_source()
    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video FPS: {fps:.2f}")

    analyzer = WaveAnalyzer(fps=fps)
    analyzer.enable_gabor_analysis = False
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
    show_texture_overlay = True
    analyzer.enable_threshold_filter = show_threshold_overlay
    show_spectrum_overlay = True
    show_cpu_profile = True
    temporal_modes = [
        {"label": "off", "enabled": False, "seconds": None, "output": "change"},
        {"label": "lp 0.5s", "enabled": True, "seconds": 0.5, "output": "lowpass"},
        {"label": "lp 2s", "enabled": True, "seconds": 2.0, "output": "lowpass"},
        {"label": "chg 0.5s", "enabled": True, "seconds": 0.5, "output": "change"},
        {"label": "chg 2s", "enabled": True, "seconds": 2.0, "output": "change"},
    ]
    temporal_mode_idx = 3

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
        "pyramid_ms",
        "lbp_ms",
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
        "pyramid_ms": 0.0,
        "lbp_ms": 0.0,
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
        "pyramid_pct": 0.0,
        "lbp_pct": 0.0,
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
    contour_motion_threshold_presets = [0.0, 15.0, 60.0]
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
            "flow_slow_interval": analyzer.flow_slow_interval,
            "flow_update_auto": analyzer.enable_auto_flow_update_interval,
            "flow_detail_mode": flow_detail_mode,
            "temporal_diff_filter": analyzer.enable_temporal_difference_filter,
            "temporal_diff_polarity": analyzer.temporal_diff_polarity,
            "flow_overlay": show_flow_overlay,
            "contour_overlay": show_contour_overlay,
            "threshold_overlay": show_threshold_overlay,
            "spectrum_overlay": show_spectrum_overlay,
            "texture_overlay": show_texture_overlay,
            "temporal_mode_label": temporal_modes[temporal_mode_idx]["label"],
            "temporal_filter": analyzer.enable_temporal_change_filter,
            "screen_blend_label": ["off", "1x", "2x"][max(0, min(2, analyzer.screen_blend_mode))],
            "pre_blur_label": ["off", "small", "large"][max(0, min(2, analyzer.blur_mode))],
            "gain_label": ["off", "-25%", "+50%", "auto"][max(0, min(3, analyzer.gain_mode))],
            "contour_motion_filter": analyzer.enable_contour_motion_filter,
            "static_contours": analyzer.show_static_contours,
            "lbp_analysis": analyzer.enable_lbp_analysis,
            "contour_motion_threshold_label": (
                "off"
                if float(analyzer.contour_motion_threshold_px) <= 0.0
                else f"{float(analyzer.contour_motion_threshold_px):.1f}px"
            ),
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
            flow_data=analysis.get("flow_data") if analysis else None,
            pyramid_data=analysis.get("pyramid_data") if analysis else None,
            wavelength_data=analysis.get("wavelength_data") if analysis else None,
            lbp_data=analysis.get("lbp_data") if analysis else None,
        )

        if show_texture_overlay and analysis:
            display = draw_lbp_overlay(display, analysis.get("lbp_data"))

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
        profile_windows["pyramid_ms"].append(analysis["timings"].get("pyramid_ms", 0.0))
        profile_windows["lbp_ms"].append(analysis["timings"].get("lbp_ms", 0.0))
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
                "pyramid_pct": avg_profile_ms["pyramid_ms"] / total_for_pct * 100.0,
                "lbp_pct": avg_profile_ms["lbp_ms"] / total_for_pct * 100.0,
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
        if key == ord("w"):
            show_texture_overlay = not show_texture_overlay
        if key == ord("c"):
            show_contour_overlay = not show_contour_overlay
        if key == ord("u"):
            if contour_motion_threshold_presets:
                contour_motion_threshold_idx = (contour_motion_threshold_idx + 1) % len(contour_motion_threshold_presets)
                analyzer.contour_motion_threshold_px = contour_motion_threshold_presets[contour_motion_threshold_idx]
        if key == ord("r"):
            show_threshold_overlay = not show_threshold_overlay
            analyzer.enable_threshold_filter = show_threshold_overlay
        if key == ord("s"):
            show_spectrum_overlay = not show_spectrum_overlay
        if key == ord("y"):
            show_cpu_profile = not show_cpu_profile
        if key == ord("m"):
            analyzer.show_mask = not analyzer.show_mask
        if key == ord("j"):
            analyzer.enable_lbp_analysis = not analyzer.enable_lbp_analysis
            profile_windows["lbp_ms"].clear()
        if key == ord("f"):
            next_idx = (flow_detail_modes.index(flow_detail_mode) + 1) % len(flow_detail_modes)
            flow_detail_mode = flow_detail_modes[next_idx]
            apply_flow_quality_for_current_mode(quality_idx)
        if key == ord("i"):
            analyzer.enable_auto_flow_update_interval = not analyzer.enable_auto_flow_update_interval
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


