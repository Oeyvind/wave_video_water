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
                # Probe available cameras (indices 0–9) and let user pick one.
                print("\nScanning for available cameras...")
                available_cameras = []
                for cam_idx in range(10):
                    cap_test = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
                    if cap_test.isOpened():
                        available_cameras.append(cam_idx)
                        cap_test.release()
                if not available_cameras:
                    print("No cameras found. Please connect a camera and try again.")
                    continue
                if len(available_cameras) == 1:
                    cam_index = available_cameras[0]
                    print(f"\nUsing live camera (index {cam_index})...\n")
                    return cam_index, Path("camera")
                print("\nAvailable cameras:")
                for cam_idx in available_cameras:
                    print(f"  {cam_idx}. Camera {cam_idx}")
                while True:
                    try:
                        cam_choice = int(input(f"Select camera index ({available_cameras[0]}-{available_cameras[-1]}): ").strip())
                        if cam_choice in available_cameras:
                            print(f"\nUsing live camera (index {cam_choice})...\n")
                            return cam_choice, Path("camera")
                        print(f"Invalid choice. Available indices: {available_cameras}")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
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


def configure_camera_capture(cap, target_width=1280, target_height=720):
    requested_width = int(target_width)
    requested_height = int(target_height)
    if hasattr(cv2, "CAP_PROP_FRAME_WIDTH"):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
    if hasattr(cv2, "CAP_PROP_FRAME_HEIGHT"):
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or requested_width)
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or requested_height)
    return requested_width, requested_height, actual_width, actual_height


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


def draw_spatial_frequency_rulers(frame, start_x=10, start_y=10, scale_factor=1.5):
    """Draw spatial frequency reference rulers in upper left corner.
    
    Three rulers (yellow, blue, purple) show characteristic spatial
    scales of the pyramid analysis bands.
    
    Args:
        frame: Image to draw on
        start_x: X position (pixels from left)
        start_y: Y position (pixels from top)
        scale_factor: Visual scaling of ruler lengths (1.0 = actual pixels shown)
    """
    # Define bands with tick spacing and colors.
    bands = [
        {
            "name": "SLo",
            "color": (0, 215, 255),  # Yellow in BGR
            "tick_step": 4,
            "label": "Fine",
        },
        {
            "name": "SMid",
            "color": (255, 0, 0),  # Blue in BGR
            "tick_step": 8,
            "label": "Coarse",
        },
        {
            "name": "SHi",
            "color": (255, 0, 255),  # Purple in BGR
            "tick_step": 16,
            "label": "Extra-coarse",
        },
    ]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    line_height = 32
    y_pos = start_y

    # Use SHi as shared reference: 4 ticks means endpoints at 0,16,32,48.
    shared_tick_count = 4
    shared_length_px = int((16 * (shared_tick_count - 1)) * scale_factor)
    
    for band in bands:
        # Draw ruler label
        label_text = f"{band['name']} ({band['label']})"
        cv2.putText(
            frame, label_text, (start_x, y_pos - 5),
            font, font_scale, band["color"], thickness, cv2.LINE_AA
        )
        
        # Draw ruler line
        x_end = start_x + shared_length_px
        cv2.line(frame, (start_x, y_pos), (x_end, y_pos), band["color"], 1)
        
        # Draw tick marks only (no numeric labels).
        for tick_idx in range(shared_tick_count):
            tick_x = start_x + int((tick_idx * band["tick_step"]) * scale_factor)
            # Tick mark
            cv2.line(frame, (tick_x, y_pos - 4), (tick_x, y_pos + 4), band["color"], 1)
        
        y_pos += line_height
    
    return frame


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
        f"quality: {quality_idx + 1}  disp 1/{switches.get('display_skip', 1)}",
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
        f"[V] flow: {switches['flow_overlay']}",
        f"[X] axial flow: {'on' if switches['flow_axial_mode'] else 'off'}",
        f"[J] LBP analysis: {'on' if switches['lbp_analysis'] else 'off'}",
        (
            "[,]/[.] order center: "
            f"{switches['lbp_order_center']:.2f}"
        ),
        (
            "[-]/['] order width: "
            f"{switches['lbp_order_width']:.2f}"
        ),
        (
            "[+]/[\\] chaos exp: "
            f"{switches['lbp_chaos_entropy_exp']:.2f}"
        ),
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
        return 0.0, 0.0, 0.0, 0.0, False

    dx = flow_block[..., 0]
    dy = flow_block[..., 1]
    mag = np.sqrt(dx * dx + dy * dy)
    active = mag > 0.12
    if not np.any(active):
        return 0.0, 0.0, 0.0, 0.0, False

    weights = mag[active]
    sum_w = float(np.sum(weights))
    sum_x = float(np.sum(dx[active] * weights))
    sum_y = float(np.sum(dy[active] * weights))
    mean_x = sum_x / max(sum_w, 1e-9)
    mean_y = sum_y / max(sum_w, 1e-9)

    direction_deg = (np.degrees(np.arctan2(sum_y, sum_x)) + 360.0) % 360.0
    strength = float(np.median(weights))
    return direction_deg, strength, mean_x, mean_y, True


def _flow_arrow_centers(frame_h, frame_w):
    q_radius = 34
    diameter = q_radius * 2
    offset = int(round(diameter * 0.6))
    left_group_h = 52 + 18 + 24 + 52 + 16 + 24

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
        if label in ("UL", "UR", "LL", "LR"):
            # UL/UR/LL/LR: circle-top and WL/S-top share one clip space,
            # preventing large offsets when one element clips before others.
            shared_top = int(np.clip(c_y - q_radius, 24, max(24, frame_h - 24 - left_group_h)))
            c_y = int(np.clip(shared_top + q_radius, q_radius + 2, frame_h - q_radius - 2))
        centers[label] = (c_x, c_y)

    centers["G"] = (int(frame_w * 0.5), int(frame_h * 0.5))
    return centers


def _draw_vertical_slider(
    frame,
    x0,
    y0,
    value,
    label,
    slider_h=68,
    slider_w=10,
    color=(80, 170, 255),
    value_max=300.0,
    label_font_scale=0.4,
):
    x0 = int(x0)
    y0 = int(y0)
    slider_h = int(max(24, slider_h))
    slider_w = int(max(6, slider_w))
    value = 0.0 if value is None else float(max(0.0, value))
    if label == "WL":
        value *= 1.3
    norm = float(np.clip(value / max(value_max, 1e-6), 0.0, 1.0))

    draw_transparent_rect(frame, x0 - 4, y0 - 16, slider_w + 8, slider_h + 22, alpha=0.52)
    base_y = y0 + slider_h
    fill_h = int(max(1, round(norm * slider_h))) if value > 0.0 else 0
    cv2.rectangle(frame, (x0, y0), (x0 + slider_w, base_y), (140, 140, 140), 1)
    if fill_h > 0:
        cv2.rectangle(frame, (x0, base_y - fill_h), (x0 + slider_w, base_y), color, -1)
    cv2.putText(
        frame,
        label,
        (x0 - 2, y0 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(max(0.2, label_font_scale)),
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )


def _draw_horizontal_slider(
    frame,
    x0,
    y0,
    value,
    label,
    slider_w=52,
    slider_h=8,
    color=(80, 210, 120),
    value_max=1.0,
    value_gain=1.0,
):
    x0 = int(x0)
def _compute_center_slit_fft_data(roi_gray):
    if roi_gray is None or getattr(roi_gray, "size", 0) == 0:
        return None

    if len(roi_gray.shape) != 2:
        return None

    h, w = roi_gray.shape[:2]
    if h <= 0 or w <= 0:
        return None

    center_y = h // 2
    slit = roi_gray[center_y, :].astype(np.float32)
    if slit.size == 0:
        return None

    waveform = np.clip(slit / 255.0, 0.0, 1.0)
    fade_len = max(1, int(round(waveform.size * 0.10)))
    if fade_len > 0:
        fade = np.ones(waveform.size, dtype=np.float32)
        edge = np.linspace(0.0, np.pi * 0.5, fade_len, dtype=np.float32)
        fade[:fade_len] = np.sin(edge) ** 2
        fade[-fade_len:] = (np.sin(edge[::-1]) ** 2)
        waveform = waveform * fade
    slit_demean = waveform - float(np.mean(waveform))
    fft_mag = np.abs(np.fft.rfft(slit_demean))
    fft_mag = fft_mag[1:] if fft_mag.size > 1 else np.asarray([], dtype=np.float32)

    if fft_mag.size > 0:
        positions = np.linspace(0.0, 1.0, fft_mag.size, dtype=np.float32)
        magnitude_sum = float(np.sum(fft_mag))
        centroid_norm = float(np.dot(positions, fft_mag) / max(magnitude_sum, 1e-9))
    else:
        centroid_norm = 0.0

    return {
        "center_y": int(center_y),
        "waveform": waveform,
        "fft_magnitude": fft_mag,
        "fft_centroid_norm": float(np.clip(centroid_norm, 0.0, 1.0)),
    }


    y0 = int(y0)
    slider_w = int(max(20, slider_w))
    slider_h = int(max(6, slider_h))
    value = 0.0 if value is None else float(max(0.0, value))
    value *= float(max(0.0, value_gain))
    norm = float(np.clip(value / max(value_max, 1e-6), 0.0, 1.0))

    draw_transparent_rect(frame, x0 - 4, y0 - 14, slider_w + 8, slider_h + 22, alpha=0.52)
    x1 = x0 + slider_w
    fill_w = int(max(1, round(norm * slider_w))) if value > 0.0 else 0
    cv2.rectangle(frame, (x0, y0), (x1, y0 + slider_h), (140, 140, 140), 1)
    if fill_w > 0:
        cv2.rectangle(frame, (x0, y0), (x0 + fill_w, y0 + slider_h), color, -1)
    cv2.putText(frame, label, (x0 - 2, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (225, 225, 225), 1, cv2.LINE_AA)


def _draw_axial_arrow(img, center, dx, dy, color, thickness, tip_length):
    """Draw a double-headed axis line through center with arrowheads at both ends."""
    p0 = (center[0] - dx, center[1] - dy)
    p1 = (center[0] + dx, center[1] + dy)
    cv2.arrowedLine(img, p0, p1, color, thickness, tipLength=tip_length)
    cv2.arrowedLine(img, p1, p0, color, thickness, tipLength=tip_length)


def _unit_tip_xy01_from_direction(direction_deg):
    """Convert arrow direction to unit-tip coordinates remapped to [0,1].

    x01: left=0, right=1
    y01: bottom=0, top=1
    """
    ux = float(np.cos(np.radians(direction_deg)))
    uy = float(-np.sin(np.radians(direction_deg)))
    x01 = float(np.clip((ux + 1.0) * 0.5, 0.0, 1.0))
    y01 = float(np.clip((uy + 1.0) * 0.5, 0.0, 1.0))
    return x01, y01


def draw_quadrant_flow_arrows(frame, flow, arrow_color=(80, 190, 255), global_color=(120, 220, 255),
                              circle_color=(220, 220, 220), global_rate_hz=None, rate_side=None,
                              axial=False, quadrant_direction_overrides=None,
                              global_direction_override=None,
                              quadrant_magnitude_overrides=None,
                              global_magnitude_override=None):
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
        direction_deg, strength, _mean_x, _mean_y, has_motion = _quadrant_vector_stats(block)
        radius = 34
        shifted_center = centers[label]

        cv2.circle(out, shifted_center, radius, circle_color, 1)

        if has_motion:
            q_color = arrow_color
            q_mag = strength
            if quadrant_magnitude_overrides is not None and label in quadrant_magnitude_overrides:
                q_mag = float(quadrant_magnitude_overrides[label])
                arrow_len = int(np.clip(16 + q_mag * 40.0, 16, 44))
            else:
                arrow_len = int(np.clip(16 + q_mag * 8.0, 16, 44))
            _dir_src = direction_deg
            if quadrant_direction_overrides is not None and label in quadrant_direction_overrides:
                _dir_src = float(quadrant_direction_overrides[label])
            _dir = _dir_src % 180.0 if axial else _dir_src
            dx = int(np.cos(np.radians(_dir)) * arrow_len)
            dy = int(np.sin(np.radians(_dir)) * arrow_len)
            if axial:
                _draw_axial_arrow(out, shifted_center, dx, dy, q_color, 2, 0.22)
            else:
                cv2.arrowedLine(out, shifted_center, (shifted_center[0] + dx, shifted_center[1] + dy), q_color, 2, tipLength=0.22)



    # Global summary arrow
    g_direction_deg, g_strength, _g_mean_x, _g_mean_y, g_has_motion = _quadrant_vector_stats(flow)
    g_center = centers["G"]
    g_radius = 42
    cv2.circle(out, g_center, g_radius, circle_color, 1)
    if g_has_motion:
        g_color = global_color
        if global_magnitude_override is not None:
            g_len = int(np.clip(20 + float(global_magnitude_override) * 48.0, 20, 56))
        else:
            g_len = int(np.clip(20 + g_strength * 10.0, 20, 56))
        _gdir_src = g_direction_deg if global_direction_override is None else float(global_direction_override)
        _gdir = _gdir_src % 180.0 if axial else _gdir_src
        g_dx = int(np.cos(np.radians(_gdir)) * g_len)
        g_dy = int(np.sin(np.radians(_gdir)) * g_len)
        if axial:
            _draw_axial_arrow(out, g_center, g_dx, g_dy, g_color, 3, 0.22)
        else:
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

    return out


def draw_adaptive_flow_arrow(frame, flow_data, axial=False):
    """Draw a single adaptive-direction arrow at the frame centre.

    Uses the quality-weighted adaptive direction from flow_data.  Drawn on a
    dedicated ring outside the fast/slow G circles so it is clearly distinct.
    Arrow length scales with direction_quality so it shrinks when unreliable.
    """
    if flow_data is None:
        return frame
    direction_deg = float(flow_data.get("adaptive_direction_deg", 0.0))
    quality = float(flow_data.get("direction_quality", 0.0))
    activity = float(flow_data.get("adaptive_activity", 0.0))
    if activity < 0.01:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    center = (cx, cy)
    a_color = (0, 220, 110)   # lime-green, visually distinct from fast (blue) and slow (dark blue)
    a_radius = 56             # slightly larger ring than G (42) to frame the arrow
    # Slightly thicker for visibility.
    a_thickness = 2
    cv2.circle(out, center, a_radius, a_color, 1)

    # Arrow length: fixed base + quality-scaled extension, capped sensibly.
    arrow_len = int(np.clip(20 + quality * 60.0, 20, 70))
    _dir = direction_deg % 180.0 if axial else direction_deg
    dx = int(np.cos(np.radians(_dir)) * arrow_len)
    dy = int(np.sin(np.radians(_dir)) * arrow_len)
    if axial:
        _draw_axial_arrow(out, center, dx, dy, a_color, a_thickness, 0.20)
    else:
        cv2.arrowedLine(out, center, (cx + dx, cy + dy), a_color, a_thickness, tipLength=0.20)

    src = flow_data.get("direction_source_label", "")
    cv2.putText(out, f"A:{src}", (cx - 18, cy - a_radius - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, a_color, 1, cv2.LINE_AA)
    return out


def _draw_lbp_triangle_widget(
    frame,
    lbp,
    tx,
    ty,
    tri_h,
    show_title=False,
    show_semantic=False,
    corner_labels=None,
    text_scale=0.26,
):
    """Draw one LBP triangle widget at top-left (tx, ty)."""
    if not lbp:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]

    tri_h = int(max(40, tri_h))
    tri_s = int(round(tri_h * 2.0 / np.sqrt(3.0)))
    tx = int(np.clip(tx, 2, max(2, w - tri_s - 2)))
    ty = int(np.clip(ty, 8, max(8, h - tri_h - 16)))

    roughness = float(lbp.get("lbp_roughness", 0.0))
    entropy = float(lbp.get("lbp_entropy", 0.0))
    uniform = float(lbp.get("lbp_uniform_ratio", 0.0))
    smooth = float(lbp.get("lbp_smooth", 0.0))
    order = float(lbp.get("lbp_order", 0.0))
    chaos = float(lbp.get("lbp_chaos", 0.0))

    p_tl = (tx, ty)
    p_tr = (tx + tri_s, ty)
    p_bot = (tx + tri_s // 2, ty + tri_h)
    p_tc = (tx + tri_s // 2, ty)
    half_s = tri_s // 2
    tri_pts = np.array([p_tl, p_tr, p_bot], dtype=np.int32)

    ov = out.copy()
    cv2.fillPoly(ov, [tri_pts], (25, 25, 25))
    cv2.addWeighted(ov, 0.58, out, 0.42, 0, out)
    cv2.polylines(out, [tri_pts], True, (140, 140, 140), 1, cv2.LINE_AA)

    bar_thick = max(2, int(round(3.0 * tri_h / 100.0)))
    vbar_w = bar_thick

    e_norm = float(np.clip(entropy / 8.0, 0.0, 1.0))
    e_len = int(round(e_norm * half_s))
    if e_len > 0:
        cv2.rectangle(out, (p_tc[0] - e_len, p_tc[1] + 1), (p_tc[0], p_tc[1] + 1 + bar_thick), (170, 90, 200), -1)

    r_len = int(round(float(np.clip(roughness, 0.0, 1.0)) * half_s))
    if r_len > 0:
        cv2.rectangle(out, (p_tc[0], p_tc[1] + 1), (p_tc[0] + r_len, p_tc[1] + 1 + bar_thick), (200, 120, 50), -1)

    u_len = int(round(float(np.clip(uniform, 0.0, 1.0)) * tri_h))
    if u_len > 0:
        bvx = p_tc[0] - vbar_w // 2
        cv2.rectangle(out, (bvx, p_tc[1]), (bvx + vbar_w, p_tc[1] + u_len), (60, 170, 210), -1)

    bary_sum = smooth + order + chaos
    if bary_sum <= 1e-6:
        smooth_n = order_n = chaos_n = 1.0 / 3.0
    else:
        smooth_n = smooth / bary_sum
        order_n = order / bary_sum
        chaos_n = chaos / bary_sum
    dot_x = int(round(chaos_n * p_tl[0] + order_n * p_tr[0] + smooth_n * p_bot[0]))
    dot_y = int(round(chaos_n * p_tl[1] + order_n * p_tr[1] + smooth_n * p_bot[1]))
    dot_r = max(2, int(round(4.0 * tri_h / 100.0)))
    cv2.circle(out, (dot_x, dot_y), dot_r, (0, 200, 80), -1, cv2.LINE_AA)
    cv2.circle(out, (dot_x, dot_y), dot_r, (0, 255, 120), 1, cv2.LINE_AA)

    f = cv2.FONT_HERSHEY_SIMPLEX
    fs = float(text_scale)
    if show_title:
        t_lbl = "LBP texture"
        (tlw, _), _ = cv2.getTextSize(t_lbl, f, fs, 1)
        cv2.putText(out, t_lbl, (tx + tri_s // 2 - tlw // 2, ty - 4), f, fs, (200, 200, 180), 1, cv2.LINE_AA)

    if show_semantic:
        c_lbl = "chaos"
        cv2.putText(out, c_lbl, (p_tl[0] - 21, p_tl[1] - 4), f, fs, (200, 200, 180), 1, cv2.LINE_AA)
        o_lbl = "order"
        (olw, _), _ = cv2.getTextSize(o_lbl, f, fs, 1)
        cv2.putText(out, o_lbl, (p_tr[0] - olw + 21, p_tr[1] - 4), f, fs, (200, 200, 180), 1, cv2.LINE_AA)
        s_lbl = "smooth"
        (slw, _), _ = cv2.getTextSize(s_lbl, f, fs, 1)
        cv2.putText(out, s_lbl, (p_bot[0] - slw // 2, p_bot[1] + 12), f, fs, (200, 200, 180), 1, cv2.LINE_AA)
    elif corner_labels is not None and len(corner_labels) == 3:
        c_lbl, o_lbl, s_lbl = [str(v) for v in corner_labels]
        cfs = fs + 0.04
        cv2.putText(out, c_lbl, (p_tl[0] - 9, p_tl[1]), f, cfs, (200, 200, 180), 1, cv2.LINE_AA)
        (olw, _), _ = cv2.getTextSize(o_lbl, f, cfs, 1)
        cv2.putText(out, o_lbl, (p_tr[0] - olw + 9, p_tr[1]), f, cfs, (200, 200, 180), 1, cv2.LINE_AA)
        (slw, _), _ = cv2.getTextSize(s_lbl, f, cfs, 1)
        cv2.putText(out, s_lbl, (p_bot[0] - slw // 2, p_bot[1] + 10), f, cfs, (200, 200, 180), 1, cv2.LINE_AA)

    lbl_y = p_tc[1] + bar_thick + max(7, int(round(9.0 * tri_h / 100.0)))
    e_lbl = f"E {entropy:.2f}"
    cv2.putText(out, e_lbl, (p_tc[0] - e_len, lbl_y), f, fs, (170, 90, 200), 1, cv2.LINE_AA)
    r_lbl = f"R {roughness:.2f}"
    (rlw, _), _ = cv2.getTextSize(r_lbl, f, fs, 1)
    cv2.putText(out, r_lbl, (p_tc[0] + r_len - rlw, lbl_y), f, fs, (200, 120, 50), 1, cv2.LINE_AA)
    u_lbl = f"U {uniform:.2f}"
    cv2.putText(out, u_lbl, (p_tc[0] + vbar_w + 2, p_tc[1] + u_len // 2 + 4), f, fs, (60, 170, 210), 1, cv2.LINE_AA)

    return out


def draw_quadrant_lbp_triangles(frame, lbp_data):
    """Draw compact LBP triangle widgets near each quadrant flow-arrow circle."""
    lbp = lbp_data or {}
    quads = lbp.get("quadrants", {})
    if not quads:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]
    centers = _flow_arrow_centers(h, w)
    q_radius = 34
    pad = 10
    tri_h = 62
    tri_s = int(round(tri_h * 2.0 / np.sqrt(3.0)))
    group_h = 52 + 18 + 24 + 52 + 16 + 24
    spatial_bar_h = 52
    centroid_stack_h = 24
    temporal_gap_h = 12
    left_temporal_shift_y = -8

    for q_label, qc in centers.items():
        if q_label == "G":
            continue
        q_lbp = quads.get(q_label, {})
        if not q_lbp:
            continue

        tx = qc[0] - tri_s // 2
        # All quadrants: triangle anchors to WL x-position (same ordering everywhere).
        tx = qc[0] - q_radius - pad - 10

        tx = int(np.clip(tx, 2, max(2, w - tri_s - 2)))
        if q_label in ("UL", "UR", "LL", "LR"):
            # UL/UR/LL/LR share one clipped anchor so all overlay elements
            # keep identical relative positions without clipping drift.
            spatial_top = int(np.clip(qc[1] - 34, 24, max(24, h - 24 - group_h)))
            temporal_top = spatial_top + spatial_bar_h + temporal_gap_h + centroid_stack_h + left_temporal_shift_y
            ty = temporal_top
        elif qc[1] < h // 2:
            ty = qc[1] + q_radius + 6
        else:
            ty = qc[1] - q_radius - tri_h - 6
        ty = int(np.clip(ty, 8, max(8, h - tri_h - 16)))

        out = _draw_lbp_triangle_widget(
            out,
            q_lbp,
            tx,
            ty,
            tri_h=tri_h,
            show_title=False,
            show_semantic=False,
            corner_labels=("C", "O", "S"),
            text_scale=0.22,
        )

    return out


def draw_lbp_overlay(frame, lbp_data):
    """Global LBP triangle, anchored below the central flow ring."""
    lbp = lbp_data or {}
    if not lbp:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]
    cx, cy = _flow_arrow_centers(h, w)["G"]

    tri_h = 77
    tri_s = int(round(tri_h * 2.0 / np.sqrt(3.0)))
    tx = cx - tri_s // 2 - 12
    ty = cy + 56 + 10 + 20
    ty = int(np.clip(ty, 8, max(8, h - tri_h - 16)))

    return _draw_lbp_triangle_widget(
        out,
        lbp,
        tx,
        ty,
        tri_h=tri_h,
        show_title=True,
        show_semantic=True,
        text_scale=0.32,
    )


# Pyramid texture band colors: yellow -> blue -> purple.
_PYRAMID_BAND_COLORS = [
    (0, 220, 255),
    (255, 0, 0),
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


def _draw_pyramid_bar_group(frame, x0, y0, bands, label, bar_w=8, bar_h=52, gap=4,
    temporal_bands=None, centroid=None, temporal_centroid=None, temporal_shift_y=0):
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

    draw_transparent_rect(frame, x0 - 4, y0 - 14, group_w + 24, group_h + 8, alpha=0.52)

    base_y = y0 + bar_h
    cv2.putText(frame, "S", (x0 + group_w + 6, base_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (205, 205, 205), 1, cv2.LINE_AA)
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
        t_y0 = base_y + 12 + temporal_offset + int(temporal_shift_y)
        t_base_y = t_y0 + temporal_h
        cv2.putText(frame, "T", (x0 + group_w + 6, t_base_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (205, 205, 205), 1, cv2.LINE_AA)
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

        # Temporal centroid rail uses provided value (renormalized) when available.
        if temporal_centroid is None:
            t_arr = np.asarray(tvals[:bar_count], dtype=np.float32)
            t_total = float(np.sum(t_arr)) + 1e-9
            t_centroid = float(np.dot(np.arange(bar_count, dtype=np.float32), t_arr) / (t_total * (bar_count - 1)))
        else:
            t_centroid = float(np.clip(temporal_centroid, 0.0, 1.0))
        tc_y0 = t_base_y + 6
        tc_h = centroid_h
        tc_mid = tc_y0 + (tc_h // 2)
        cv2.line(frame, (x0, tc_mid), (x0 + group_w, tc_mid), (130, 130, 130), 1, cv2.LINE_AA)
        for t in range(bar_count):
            tx = x0 + int(round((group_w - 1) * (t / max(1, bar_count - 1))))
            cv2.line(frame, (tx, tc_mid - 2), (tx, tc_mid + 2), (90, 90, 90), 1, cv2.LINE_AA)
        tc_marker_x = x0 + int(round(float(np.clip(t_centroid, 0.0, 1.0)) * (group_w - 1)))
        cv2.rectangle(frame, (tc_marker_x - 2, tc_y0), (tc_marker_x + 2, tc_y0 + tc_h), (180, 180, 180), -1)


def draw_pyramid_texture_bars(frame, pyramid_data, wavelength_data=None, activity_data=None):
    if not isinstance(pyramid_data, dict):
        return frame

    def _scale_pyramid_vals(vals):
        if vals is None:
            return None
        return [float(np.clip(float(v) * 1.3, 0.0, 1.0)) for v in vals]

    out = frame
    h, w = out.shape[:2]
    g = _scale_pyramid_vals(pyramid_data.get("global_bands", [0.0, 0.0, 0.0]))
    g_t = _scale_pyramid_vals(pyramid_data.get("temporal_band_activity", [0.0, 0.0, 0.0]))
    g_ctr = float(pyramid_data.get("scale_centroid_renorm", pyramid_data.get("scale_centroid", 0.0)))
    g_t_ctr = float(pyramid_data.get("temporal_scale_centroid", 0.0))
    q = pyramid_data.get("quadrant_bands", {})
    q_t = pyramid_data.get("quadrant_temporal_bands", {})
    q_ctr = pyramid_data.get("quadrant_scale_centroids_renorm", pyramid_data.get("quadrant_scale_centroids", {}))
    q_t_ctr = pyramid_data.get("quadrant_temporal_scale_centroids", {})
    wd = wavelength_data or {}
    q_wl = wd.get("quadrants", {})
    g_wl = wd.get("wavelength_px")
    ad = activity_data or {}
    q_act = ad.get("quadrant_activity", {}) if isinstance(ad, dict) else {}
    g_act = float(ad.get("global_activity", 0.0)) if isinstance(ad, dict) else 0.0
    q_dft = ad.get("quadrant_components", {}) if isinstance(ad, dict) else {}
    g_dft = ad.get("global_components", {}) if isinstance(ad, dict) else {}

    centers = _flow_arrow_centers(h, w)

    bar_count = len(_PYRAMID_BAND_COLORS)
    group_w = (8 * bar_count) + (4 * max(0, bar_count - 1))
    # spatial(52) + label(18) + s_centroid(24) + T_bars(52) + T_gap(16) + t_centroid(24)
    group_h = 52 + 18 + 24 + 52 + 16 + 24
    group_h_global = group_h
    q_radius = 34
    g_radius = 42
    pad = 10
    left_temporal_shift_y = -8

    for label in ("UL", "UR", "LL", "LR"):
        cx, cy = centers[label]
        # Same ordering in all quadrants: WL (left) -> flow circle -> S/T bars (right).
        x = cx + q_radius + pad
        wl_x = cx - q_radius - pad - 10
        y = cy - (group_h // 2)
        wl_y = cy - 34
        if label in ("UL", "UR", "LL", "LR"):
            # UL/UR/LL/LR: use one clipped top reference for both S bars and WL.
            left_top = int(np.clip(cy - 34, 24, max(24, h - 24 - group_h)))
            y = left_top
            wl_y = left_top
        x = int(np.clip(x, 14, max(14, w - 14 - group_w)))
        y = int(np.clip(y, 24, max(24, h - 24 - group_h)))
        _draw_pyramid_bar_group(out, x, y, _scale_pyramid_vals(q.get(label, [0.0] * bar_count)), "",
            temporal_bands=_scale_pyramid_vals(q_t.get(label)), centroid=q_ctr.get(label),
            temporal_centroid=q_t_ctr.get(label),
            temporal_shift_y=(left_temporal_shift_y if label in ("UL", "UR", "LL", "LR") else 0))
        wl_val = (q_wl.get(label) or {}).get("wavelength_px")
        act_val = float((q_act.get(label) if isinstance(q_act, dict) else 0.0) or 0.0)
        wl_y = int(np.clip(wl_y, 24, max(24, h - 24 - 68)))
        wl_x = int(np.clip(wl_x, 10, max(10, w - 10 - 10)))
        act_w = 10
        act_h = 68
        act_x = int(np.clip(wl_x - act_w - 8, 10, max(10, w - 10 - act_w)))
        act_y = int(np.clip(wl_y, 24, max(24, h - 24 - act_h)))

        dft_vals = q_dft.get(label, {}) if isinstance(q_dft, dict) else {}
        dft_w = 7
        dft_gap = 3
        dft_group_w = (3 * dft_w) + (2 * dft_gap)
        dft_x0 = int(np.clip(act_x - 6 - dft_group_w, 10, max(10, w - 10 - dft_group_w)))
        _draw_vertical_slider(
            out,
            dft_x0,
            act_y,
            float(dft_vals.get("diff", 0.0)) * 6.0,
            label="d",
            slider_h=act_h,
            slider_w=dft_w,
            color=(70, 190, 255),
            value_max=1.0,
            label_font_scale=0.28,
        )
        _draw_vertical_slider(
            out,
            dft_x0 + dft_w + dft_gap,
            act_y,
            float(dft_vals.get("flow", 0.0)) * 3.0,
            label="f",
            slider_h=act_h,
            slider_w=dft_w,
            color=(80, 210, 120),
            value_max=1.0,
            label_font_scale=0.28,
        )
        _draw_vertical_slider(
            out,
            dft_x0 + 2 * (dft_w + dft_gap),
            act_y,
            float(dft_vals.get("tex", 0.0)) * 1.0,
            label="t",
            slider_h=act_h,
            slider_w=dft_w,
            color=(210, 130, 250),
            value_max=1.0,
            label_font_scale=0.28,
        )
        _draw_vertical_slider(out, act_x, act_y, act_val * 3.0, label="A", slider_h=act_h, slider_w=act_w, color=(80, 210, 120), value_max=1.0)
        _draw_vertical_slider(out, wl_x, wl_y, wl_val, label="WL", slider_h=68, slider_w=10, color=(30, 140, 255), value_max=300.0)

    g_cx, g_cy = centers["G"]
    g_bar_x = int(np.clip(g_cx + g_radius + pad + 12, 14, max(14, w - 14 - group_w)))
    g_bar_y = int(np.clip(g_cy + 14 - 15, 24, max(24, h - 24 - group_h_global)))
    _draw_pyramid_bar_group(out, g_bar_x, g_bar_y, g, "", temporal_bands=g_t, centroid=g_ctr, temporal_centroid=g_t_ctr)
    g_wl_x = int(np.clip(g_cx - g_radius - pad - 10 - 12, 10, max(10, w - 10 - 10)))
    g_wl_y = int(np.clip(g_cy + 14 - 15, 24, max(24, h - 24 - 68)))
    g_act_w = 10
    g_act_h = 68
    g_act_x = int(np.clip(g_wl_x - g_act_w - 8, 10, max(10, w - 10 - g_act_w)))
    g_act_y = int(np.clip(g_wl_y, 24, max(24, h - 24 - g_act_h)))

    g_d = float(g_dft.get("diff", 0.0)) if isinstance(g_dft, dict) else 0.0
    g_f = float(g_dft.get("flow", 0.0)) if isinstance(g_dft, dict) else 0.0
    g_t = float(g_dft.get("tex", 0.0)) if isinstance(g_dft, dict) else 0.0
    g_dft_w = 7
    g_dft_gap = 3
    g_dft_group_w = (3 * g_dft_w) + (2 * g_dft_gap)
    g_dft_x0 = int(np.clip(g_act_x - 6 - g_dft_group_w, 10, max(10, w - 10 - g_dft_group_w)))
    _draw_vertical_slider(
        out,
        g_dft_x0,
        g_act_y,
        g_d * 6.0,
        label="d",
        slider_h=g_act_h,
        slider_w=g_dft_w,
        color=(70, 190, 255),
        value_max=1.0,
        label_font_scale=0.28,
    )
    _draw_vertical_slider(
        out,
        g_dft_x0 + g_dft_w + g_dft_gap,
        g_act_y,
        g_f * 3.0,
        label="f",
        slider_h=g_act_h,
        slider_w=g_dft_w,
        color=(80, 210, 120),
        value_max=1.0,
        label_font_scale=0.28,
    )
    _draw_vertical_slider(
        out,
        g_dft_x0 + 2 * (g_dft_w + g_dft_gap),
        g_act_y,
        g_t * 1.0,
        label="t",
        slider_h=g_act_h,
        slider_w=g_dft_w,
        color=(210, 130, 250),
        value_max=1.0,
        label_font_scale=0.28,
    )
    _draw_vertical_slider(out, g_act_x, g_act_y, g_act * 3.0, label="A", slider_h=g_act_h, slider_w=g_act_w, color=(80, 210, 120), value_max=1.0)
    _draw_vertical_slider(out, g_wl_x, g_wl_y, g_wl, label="WL", slider_h=68, slider_w=10, color=(30, 140, 255), value_max=300.0)
    return out


def draw_slit_fft_overlay(frame, slit_fft_data):
    """Draw the center-slit waveform and its FFT magnitude on the frame.

    Waveform: yellow, zero-line at the slit row, amplitude up to 200 px upward.
    FFT graph: orange, placed at center-top spanning the middle third of the frame.
    """
    if not isinstance(slit_fft_data, dict):
        return frame

    out = frame.copy()
    h, w = out.shape[:2]
    center_y = int(slit_fft_data.get("center_y", h // 2))
    waveform = np.asarray(slit_fft_data.get("waveform", []), dtype=np.float32)
    fft_mag = np.asarray(slit_fft_data.get("fft_magnitude", []), dtype=np.float32)
    slit_len = int(waveform.shape[0])

    max_amp = 200
    yellow = (0, 255, 255)
    orange = (0, 165, 255)

    # --- Waveform ---
    # Draw a dim zero-reference line at the slit row.
    cv2.line(out, (0, center_y), (w - 1, center_y), (60, 60, 0), 1)

    wv_pts = []
    for i in range(slit_len):
        x = int(round(i * (w - 1) / max(slit_len - 1, 1)))
        y = center_y - int(float(np.clip(waveform[i], 0.0, 1.0)) * max_amp)
        wv_pts.append([x, y])

    if len(wv_pts) > 1:
        wv_arr = np.array(wv_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [wv_arr], False, yellow, 1, cv2.LINE_AA)

    # --- FFT ---
    fft_x0 = w // 3
    fft_x1 = (2 * w) // 3
    fft_w_px = fft_x1 - fft_x0
    fft_h_px = 150
    fft_top = 10

    n_bins = fft_mag.shape[0]

    fft_peak = float(np.max(fft_mag)) if n_bins > 0 else 1.0
    if fft_peak < 1e-6:
        fft_peak = 1.0

    # Semi-transparent background for readability.
    draw_transparent_rect(out, fft_x0 - 2, fft_top - 2, fft_w_px + 4, fft_h_px + 4, alpha=0.45)

    fft_pts = []
    for i in range(n_bins):
        x = fft_x0 + int(round(i * (fft_w_px - 1) / max(n_bins - 1, 1)))
        norm = float(fft_mag[i]) / fft_peak
        y = fft_top + fft_h_px - int(norm * fft_h_px)
        y = int(np.clip(y, fft_top, fft_top + fft_h_px))
        fft_pts.append([x, y])

    if len(fft_pts) > 1:
        fft_arr = np.array(fft_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [fft_arr], False, orange, 1, cv2.LINE_AA)

    centroid_norm = float(np.clip(slit_fft_data.get("fft_centroid_norm", 0.0), 0.0, 1.0))
    centroid_x = fft_x0 + int(round(centroid_norm * (fft_w_px - 1)))
    centroid_y = fft_top + fft_h_px
    cv2.circle(out, (centroid_x, centroid_y), 4, (0, 0, 255), -1, cv2.LINE_AA)

    # Baseline for FFT area.
    cv2.line(out, (fft_x0, fft_top + fft_h_px), (fft_x1, fft_top + fft_h_px), (100, 80, 0), 1)

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
    texture_overlay,
    axial_flow=False,
    lbp_data=None,
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
            cv2.drawContours(out, blob_contours, -1, (30, 250, 70), 1)
            # Highlight the largest blob-like contour in red
            largest_idx = int(np.argmax([cv2.contourArea(c) for c in blob_contours]))
            cv2.drawContours(out, blob_contours, largest_idx, (0, 0, 255), 1)

        straight_line_contours = analysis.get("static_contours", [])
        if straight_line_contours:
            cv2.drawContours(out, straight_line_contours, -1, (0, 220, 255), 1)

    if flow_overlay != "off":
        flow_data = analysis.get("flow_data") or {}
        fast_qm = flow_data.get("quadrant_fast_metrics") or {}
        slow_qm = flow_data.get("quadrant_slow_metrics") or {}
        fast_q_dirs = {q: float((fast_qm.get(q) or {}).get("direction_deg", 0.0)) for q in ("UL", "UR", "LL", "LR")}
        slow_q_dirs = {q: float((slow_qm.get(q) or {}).get("direction_deg", 0.0)) for q in ("UL", "UR", "LL", "LR")}
        fast_q_mags = {q: float((fast_qm.get(q) or {}).get("activity", 0.0)) for q in ("UL", "UR", "LL", "LR")}
        slow_q_mags = {q: float((slow_qm.get(q) or {}).get("activity", 0.0)) for q in ("UL", "UR", "LL", "LR")}

        fast_rate_hz = float(analyzer.fps) / max(1, int(analyzer.flow_update_interval))
        slow_rate_hz = float(analyzer.fps) / max(1, int(analyzer.flow_slow_interval))
        fast_arrow_color = _rate_hz_to_color(fast_rate_hz)
        slow_arrow_color = _rate_hz_to_color(slow_rate_hz)

        # Fast flow: color keyed by fast analysis rate
        if flow_overlay == "full":
            out = draw_flow_overlay(out, analyzer.last_flow, color=(30, 180, 255), disp_scale=2.0)
        out = draw_quadrant_flow_arrows(out, analyzer.last_flow,
                                        arrow_color=fast_arrow_color, global_color=fast_arrow_color,
                                        circle_color=(200, 200, 200),
                                        global_rate_hz=fast_rate_hz, rate_side="right",
                                        axial=axial_flow,
                                        quadrant_direction_overrides=fast_q_dirs,
                                        global_direction_override=flow_data.get("fast_direction_deg"),
                                        quadrant_magnitude_overrides=fast_q_mags,
                                        global_magnitude_override=flow_data.get("fast_activity"),
                                        )
        # Slow flow: color keyed by slow analysis rate
        slow_disp_scale = 2.0 / max(1, analyzer.flow_slow_interval)
        if flow_overlay == "full":
            out = draw_flow_overlay(out, analyzer.last_flow_slow, color=(30, 100, 255), disp_scale=slow_disp_scale)
        out = draw_quadrant_flow_arrows(out, analyzer.last_flow_slow,
                                        arrow_color=slow_arrow_color, global_color=slow_arrow_color,
                                        circle_color=(160, 160, 160),
                                        global_rate_hz=slow_rate_hz, rate_side="left",
                                        axial=axial_flow,
                                        quadrant_direction_overrides=slow_q_dirs,
                                        global_direction_override=flow_data.get("slow_direction_deg"),
                                        quadrant_magnitude_overrides=slow_q_mags,
                                        global_magnitude_override=flow_data.get("slow_activity"),
                                        )
        if texture_overlay:
            # Per-quadrant LBP triangles: shown with the texture overlay toggle (W).
            out = draw_quadrant_lbp_triangles(out, lbp_data)

    if spectrum_overlay:
        out = draw_pyramid_texture_bars(
            out,
            analysis.get("pyramid_data"),
            analysis.get("wavelength_data"),
            analysis.get("activity_data"),
        )

    out = draw_slit_fft_overlay(out, analysis.get("slit_fft_data"))

    return out


def main():
    video_source, source_path = select_video_source()
    if isinstance(video_source, int):
        cap = cv2.VideoCapture(video_source, cv2.CAP_DSHOW)
        requested_width, requested_height, actual_width, actual_height = configure_camera_capture(
            cap,
            target_width=1280,
            target_height=720,
        )
        if (actual_width, actual_height) != (requested_width, requested_height):
            print(
                f"Requested camera resolution: {requested_width}x{requested_height} "
                f"(backend reported {actual_width}x{actual_height}; frames will be downsampled)"
            )
        else:
            print(f"Requested camera resolution: {requested_width}x{requested_height}")
    else:
        cap = cv2.VideoCapture(video_source)
        requested_width = 1280
        requested_height = 720
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

    quality_idx = 0
    analyzer.set_quality(**quality_presets[quality_idx])
    apply_flow_quality_for_current_mode(quality_idx)
    # Display rendering every N analysis frames — reduces overlay CPU at lower quality levels.
    display_skip_intervals = [1, 2, 3, 4]
    _display_frame_counter = 0
    _last_display = None
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
    flow_mode = "arrows"  # "off" | "arrows" | "full"
    flow_axial_mode = False
    show_contour_overlay = False
    show_threshold_overlay = False
    show_texture_overlay = True
    analyzer.enable_threshold_filter = show_threshold_overlay
    show_spectrum_overlay = True
    show_cpu_profile = False
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
        result = get_frame(cap, loop=loop_video, target_size=(requested_width, requested_height))
        cap_t1 = time.perf_counter()
        profile_windows["capture_ms"].append((cap_t1 - cap_t0) * 1000.0)
        if result is None:
            break

        frame, gray = result
        frame_timestamps.append(time.perf_counter())
        analysis = analyzer.analyze(gray)
        analysis["slit_fft_data"] = _compute_center_slit_fft_data(analysis.get("roi_gray"))
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
            "flow_overlay": flow_mode,
            "display_skip": display_skip_intervals[quality_idx],
            "flow_axial_mode": flow_axial_mode,
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
            "lbp_order_center": float(getattr(analyzer, "lbp_order_center", 0.5)),
            "lbp_order_width": float(getattr(analyzer, "lbp_order_width", 0.5)),
            "lbp_chaos_entropy_exp": float(getattr(analyzer, "lbp_chaos_entropy_exp", 1.0)),
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
        _display_frame_counter = (_display_frame_counter + 1) % display_skip_intervals[quality_idx]
        _do_render = (_display_frame_counter == 0) or (_last_display is None)
        render_t0 = time.perf_counter()
        if _do_render:
            display = make_display_frame(
                frame,
                analysis,
                mode_name,
                analyzer,
                show_threshold_overlay,
                show_contour_overlay,
                flow_mode,
                show_spectrum_overlay,
                show_texture_overlay,
                axial_flow=flow_axial_mode,
                lbp_data=analysis.get("lbp_data") if analysis else None,
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

            # Draw spatial frequency reference rulers
            display = draw_spatial_frequency_rulers(display, start_x=10, start_y=10, scale_factor=1.5)

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

            _last_display = display

        render_t1 = time.perf_counter()
        profile_windows["render_ms"].append((render_t1 - render_t0) * 1000.0 if _do_render else 0.0)

        cv2.imshow("Wave Analyzer", _last_display)

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
            flow_mode = {"off": "arrows", "arrows": "full", "full": "off"}[flow_mode]
        if key == ord("x"):
            flow_axial_mode = not flow_axial_mode
            analyzer.flow_axial_mode = flow_axial_mode
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
        if key == ord(","):
            analyzer.set_lbp_compound_tuning(order_center=analyzer.lbp_order_center - 0.02)
        if key == ord("."):
            analyzer.set_lbp_compound_tuning(order_center=analyzer.lbp_order_center + 0.02)
        if key == ord("-"):
            analyzer.set_lbp_compound_tuning(order_width=analyzer.lbp_order_width - 0.02)
        if key == ord("'"):
            analyzer.set_lbp_compound_tuning(order_width=analyzer.lbp_order_width + 0.02)
        if key == ord("+"):
            analyzer.set_lbp_compound_tuning(chaos_entropy_exp=analyzer.lbp_chaos_entropy_exp - 0.05)
        if key == ord("\\"):
            analyzer.set_lbp_compound_tuning(chaos_entropy_exp=analyzer.lbp_chaos_entropy_exp + 0.05)
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


