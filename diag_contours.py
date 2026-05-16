"""Quick contour diagnostic: show what the preprocessing pipeline produces
for a mid-video frame of each test file, trying several approaches."""
import sys
import cv2
import numpy as np

VIDEOS = [
    r"C:\Projects\efx_experiments\wave_video_files\Brattøra_4.mp4",
    r"C:\Projects\efx_experiments\wave_video_files\Capetown_1.mp4",
]
MIN_AREA = 20.0

def grab_frame(path, frac=0.4):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read {path}")
    return frame

def analyze(frame, label):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- current approach: findContours on raw grayscale ---
    c_raw, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_raw = [c for c in c_raw if cv2.contourArea(c) >= MIN_AREA]

    # --- Otsu binary threshold ---
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    c_otsu, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_otsu = [c for c in c_otsu if cv2.contourArea(c) >= MIN_AREA]

    # --- Adaptive threshold ---
    adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 21, 3)
    c_adapt, _ = cv2.findContours(adapt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_adapt = [c for c in c_adapt if cv2.contourArea(c) >= MIN_AREA]

    # --- Canny edges ---
    canny = cv2.Canny(blur, 40, 120)
    canny_d = cv2.dilate(canny, np.ones((2, 2), np.uint8), iterations=1)
    c_canny, _ = cv2.findContours(canny_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_canny = [c for c in c_canny if cv2.contourArea(c) >= MIN_AREA]

    # --- Canny with lower thresholds ---
    canny_lo = cv2.Canny(blur, 15, 50)
    canny_lo_d = cv2.dilate(canny_lo, np.ones((2, 2), np.uint8), iterations=1)
    c_canny_lo, _ = cv2.findContours(canny_lo_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_canny_lo = [c for c in c_canny_lo if cv2.contourArea(c) >= MIN_AREA]

    print(f"\n=== {label} ===")
    print(f"  raw grayscale findContours : {len(c_raw):4d} contours")
    print(f"  Otsu binary               : {len(c_otsu):4d} contours")
    print(f"  Adaptive threshold        : {len(c_adapt):4d} contours")
    print(f"  Canny(40,120)+dilate      : {len(c_canny):4d} contours")
    print(f"  Canny(15,50)+dilate       : {len(c_canny_lo):4d} contours")

    # Save side-by-side visual
    h, w = gray.shape
    def to_bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img

    def draw_c(img, contours, color):
        out = to_bgr(img.copy())
        cv2.drawContours(out, contours, -1, color, 1)
        return out

    row1 = np.hstack([
        draw_c(cv2.resize(frame, (320, 240)), [], (0,0,0)),
        draw_c(cv2.resize(blur, (320, 240)), c_raw[:200], (0, 255, 0)),
        draw_c(cv2.resize(otsu, (320, 240)), c_otsu[:200], (0, 200, 255)),
    ])
    row2 = np.hstack([
        draw_c(cv2.resize(adapt, (320, 240)), c_adapt[:200], (255, 180, 0)),
        draw_c(cv2.resize(canny, (320, 240)), c_canny[:200], (0, 255, 200)),
        draw_c(cv2.resize(canny_lo, (320, 240)), c_canny_lo[:200], (180, 0, 255)),
    ])
    grid = np.vstack([row1, row2])

    def add_label(img, texts, y_start=15):
        for i, t in enumerate(texts):
            cv2.putText(img, t, (5, y_start + i*14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,0), 1)
    add_label(row1, ["original", "raw gray (current)", "Otsu binary"])
    add_label(row2, ["adaptive thresh", "Canny(40,120)+dil", "Canny(15,50)+dil"])

    out_path = f"diag_{label.replace(' ','_')}.png"
    cv2.imwrite(out_path, grid)
    print(f"  Saved: {out_path}")
    return gray, blur, otsu, adapt, canny, canny_lo

for path in VIDEOS:
    name = path.split("\\")[-1].replace(".mp4", "")
    try:
        frame = grab_frame(path)
        analyze(frame, name)
    except Exception as e:
        print(f"ERROR {name}: {e}")

print("\nDone.")
