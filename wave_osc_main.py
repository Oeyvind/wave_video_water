"""Send a circular brightness scan of a video surface to Csound over OSC."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from pythonosc.udp_client import SimpleUDPClient

from video_capture import get_frame


OSC_PATH = "/wave/circle/chunk"
OSC_HOST = "127.0.0.1"
OSC_PORT = 8101
SAMPLE_COUNT = 512
CHUNK_SIZE = 16
DISPLAY_MODES = ("filtered", "raw")
TEMPORAL_MODES = (
    ("off", False, None, "change"),
    ("lp 0.5s", True, 0.5, "lowpass"),
    ("lp 2s", True, 2.0, "lowpass"),
    ("chg 0.5s", True, 0.5, "change"),
    ("chg 2s", True, 2.0, "change"),
)


def select_video_source():
    candidates = [
        Path("C:/Projects/efx_experiments/wave_video_files"),
        Path("C:/Cabbage_VST/CabbageEfx/wave_video_files"),
        Path.cwd(),
    ]
    video_dir = next((path for path in candidates if path.exists()), Path.cwd())
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    video_files = sorted(path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions)
    print("\n0. Live camera")
    for index, path in enumerate(video_files, start=1):
        print(f"{index}. {path.name}")

    while True:
        try:
            choice = int(input("Select source: ").strip())
        except ValueError:
            print("Enter a number.")
            continue
        if choice == 0:
            cameras = []
            for camera_index in range(10):
                probe = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
                if probe.isOpened():
                    cameras.append(camera_index)
                probe.release()
            if not cameras:
                print("No cameras found.")
                continue
            if len(cameras) == 1:
                return cameras[0], Path("camera")
            print(f"Available cameras: {cameras}")
            try:
                camera_index = int(input("Camera index: ").strip())
            except ValueError:
                continue
            if camera_index in cameras:
                return camera_index, Path("camera")
        elif 1 <= choice <= len(video_files):
            return str(video_files[choice - 1]), video_files[choice - 1]
        else:
            print("Invalid source.")


def scan_circle(gray_frame, center_x, center_y, radius, sample_count=SAMPLE_COUNT):
    """Sample normalized brightness clockwise around a circular path."""
    height, width = gray_frame.shape[:2]
    angles = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False, dtype=np.float32)
    x = np.clip(np.rint(center_x + radius * np.cos(angles)).astype(np.int32), 0, width - 1)
    y = np.clip(np.rint(center_y + radius * np.sin(angles)).astype(np.int32), 0, height - 1)
    return gray_frame[y, x].astype(np.float32) / 255.0


def send_waveform(client, waveform, frame_id):
    waveform = np.clip(np.asarray(waveform, dtype=np.float32).reshape(-1), 0.0, 1.0)
    if waveform.size != SAMPLE_COUNT:
        raise ValueError(f"Expected {SAMPLE_COUNT} samples, got {waveform.size}")
    chunk_count = SAMPLE_COUNT // CHUNK_SIZE
    for chunk_index in range(chunk_count):
        start = chunk_index * CHUNK_SIZE
        payload = [int(frame_id), chunk_index, chunk_count]
        payload.extend(float(value) for value in waveform[start:start + CHUNK_SIZE])
        client.send_message(OSC_PATH, payload)


def apply_filters(gray_frame, state, fps):
    """Apply the global image filters used by main.py."""
    source = gray_frame
    _label, temporal_enabled, seconds, output_mode = TEMPORAL_MODES[state["temporal_mode"]]
    if temporal_enabled:
        alpha = 1.0 / max(1.0, fps * seconds)
        source_float = state["temporal_float"]
        if source_float is None or source_float.shape != source.shape:
            source_float = source.astype(np.float32)
        else:
            cv2.accumulateWeighted(source, source_float, alpha)
        state["temporal_float"] = source_float
        lowpass = cv2.convertScaleAbs(source_float)
        if output_mode == "lowpass":
            source = lowpass
        else:
            source = cv2.subtract(source, lowpass)
    else:
        state["temporal_float"] = None

    if state["temporal_diff"]:
        previous = state["previous_diff"]
        if previous is None or previous.shape != source.shape:
            source = np.zeros_like(source)
        elif state["diff_polarity"] == "positive":
            source = cv2.subtract(source, previous)
        elif state["diff_polarity"] == "negative":
            source = cv2.subtract(previous, source)
        else:
            source = cv2.absdiff(source, previous)
        state["previous_diff"] = gray_frame.copy()
    else:
        state["previous_diff"] = None

    if state["screen_blend"]:
        values = source.astype(np.float32) / 255.0
        values = 1.0 - (1.0 - values) ** (2 ** state["screen_blend"])
        source = np.clip(values * 255.0, 0, 255).astype(np.uint8)

    gain_mode = state["gain_mode"]
    if gain_mode == 1:
        source = cv2.convertScaleAbs(source, alpha=0.75)
    elif gain_mode == 2:
        source = cv2.convertScaleAbs(source, alpha=1.50)
    elif gain_mode == 3:
        nonzero = source[source > 0]
        p95 = float(np.percentile(nonzero, 95)) if nonzero.size else 1.0
        source = cv2.convertScaleAbs(source, alpha=min(8.0, 220.0 / max(p95, 1.0)))

    blur_kernels = ((5, 5), (9, 9), (15, 15))
    if state["blur_mode"]:
        source = cv2.GaussianBlur(source, blur_kernels[state["blur_mode"] - 1], 0)
    return source


def draw_legend(frame, state):
    temporal_label = TEMPORAL_MODES[state["temporal_mode"]][0]
    gain_label = ("off", "-25%", "+50%", "auto")[state["gain_mode"]]
    blur_label = ("off", "small", "large", "xlarge")[state["blur_mode"]]
    screen_label = ("off", "1x", "2x")[state["screen_blend"]]
    lines = [
        "Keys",
        f"[D] mode: {DISPLAY_MODES[state['display_mode']]}",
        f"[T] temporal: {temporal_label}",
        f"[H] temporal diff: {'on' if state['temporal_diff'] else 'off'}",
        f"[G] diff polarity: {state['diff_polarity']}",
        f"[E] screen: {screen_label}",
        f"[A] gain: {gain_label}",
        f"[B] blur: {blur_label}",
        "arrows center  +/- radius",
        "[Q] quit",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    pad = 8
    line_height = 20
    widths = [cv2.getTextSize(line, font, scale, 1)[0][0] for line in lines]
    box_w = max(widths) + pad * 2
    box_h = line_height * len(lines) + pad * 2
    x0 = max(0, frame.shape[1] - box_w - 10)
    y0 = 10
    overlay = frame[y0:y0 + box_h, x0:x0 + box_w].copy()
    cv2.rectangle(overlay, (0, 0), (box_w - 1, box_h - 1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame[y0:y0 + box_h, x0:x0 + box_w], 0.45, 0, frame[y0:y0 + box_h, x0:x0 + box_w])
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (x0 + pad, y0 + pad + 15 + index * line_height), font, scale, (235, 235, 235), 1, cv2.LINE_AA)
    return frame


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center-x", type=float, default=None, help="Circle center X in pixels")
    parser.add_argument("--center-y", type=float, default=None, help="Circle center Y in pixels")
    parser.add_argument("--radius", type=float, default=None, help="Circle radius in pixels")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    source, source_path = select_video_source()
    capture = cv2.VideoCapture(source, cv2.CAP_DSHOW) if isinstance(source, int) else cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    client = SimpleUDPClient(OSC_HOST, OSC_PORT)
    center_x, center_y, radius = args.center_x, args.center_y, args.radius
    frame_id = 0
    filter_state = {
        "display_mode": 0,
        "temporal_mode": 3,
        "temporal_diff": False,
        "diff_polarity": "positive",
        "screen_blend": 1,
        "gain_mode": 0,
        "blur_mode": 1,
        "temporal_float": None,
        "previous_diff": None,
    }
    print(f"Source: {source_path}; OSC: {OSC_HOST}:{OSC_PORT}{OSC_PATH}")
    print("Keys: arrows move center, D/T/H/G/E/A/B filters, +/- radius, q quits")

    try:
        while True:
            result = get_frame(capture, loop=not isinstance(source, int), target_size=(args.width, args.height))
            if result is None:
                break
            frame, gray_frame = result
            height, width = gray_frame.shape[:2]
            center_x = width * 0.5 if center_x is None else center_x
            center_y = height * 0.5 if center_y is None else center_y
            radius = min(width, height) * 0.25 if radius is None else radius
            radius = float(np.clip(radius, 1.0, min(width, height) * 0.5))
            filtered_gray = apply_filters(gray_frame, filter_state, args.fps)
            send_waveform(client, scan_circle(filtered_gray, center_x, center_y, radius), frame_id)
            frame_id += 1

            display_gray = filtered_gray if filter_state["display_mode"] == 0 else gray_frame
            display = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2BGR)
            center = (int(round(center_x)), int(round(center_y)))
            cv2.circle(display, center, int(round(radius)), (0, 220, 255), 2, cv2.LINE_AA)
            display = draw_legend(display, filter_state)
            cv2.imshow("Circular Wave Scan", display)
            key = cv2.waitKeyEx(max(1, int(round(1000.0 / max(args.fps, 1.0)))))
            if key == ord("q"):
                break
            if key == ord("d"):
                filter_state["display_mode"] = (filter_state["display_mode"] + 1) % len(DISPLAY_MODES)
            elif key == ord("t"):
                filter_state["temporal_mode"] = (filter_state["temporal_mode"] + 1) % len(TEMPORAL_MODES)
                filter_state["temporal_float"] = None
            elif key == ord("h"):
                filter_state["temporal_diff"] = not filter_state["temporal_diff"]
                filter_state["previous_diff"] = None
            elif key == ord("g"):
                polarities = ("positive", "negative", "both")
                index = polarities.index(filter_state["diff_polarity"])
                filter_state["diff_polarity"] = polarities[(index + 1) % len(polarities)]
            elif key == ord("e"):
                filter_state["screen_blend"] = (filter_state["screen_blend"] + 1) % 3
            elif key == ord("a"):
                filter_state["gain_mode"] = (filter_state["gain_mode"] + 1) % 4
            elif key == ord("b"):
                filter_state["blur_mode"] = (filter_state["blur_mode"] + 1) % 4
            elif key in (81, 2424832):
                center_x -= 5
            elif key in (83, 2555904):
                center_x += 5
            elif key in (82, 2490368):
                center_y -= 5
            elif key in (84, 2621440):
                center_y += 5
            elif key in (ord("+"), ord("=")):
                radius += 5
            elif key in (ord("-"), ord("_")):
                radius -= 5
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()