import cv2
from video_capture import get_frame
from wave_analysis import analyze_direction, analyze_frequencies
from osc_sender import send_wave_data
from spectrum_plot import render_spectrum_overlay

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("../wave_video_files/dokkpark_2025_10.mp4")
step = False
show_spectrogram = True
show_summary = True

fps = cap.get(cv2.CAP_PROP_FPS) or 30
print(f"Video FPS: {fps:.2f}")
prev_gray = None
intensity_series = []
max_len = 128

while True:
    result = get_frame(cap)
    if result is None:
        break
    frame, gray = result
    gray_small = cv2.resize(gray, (160, 120))

    roi = gray[100:110, :]
    avg_intensity = roi.mean()
    intensity_series.append(avg_intensity)
    if len(intensity_series) > max_len:
        intensity_series.pop(0)

    if prev_gray is not None and len(intensity_series) >= max_len:
        direction = analyze_direction(prev_gray, gray_small)
        freqs = analyze_frequencies(intensity_series, fps)
        send_wave_data(freqs, direction)
        render_spectrum_overlay(
            frame,
            freqs["xf"],
            freqs["yf"],
            {
                "low": freqs["low"],
                "mid": freqs["mid"],
                "high": freqs["high"],
            },
            show_spectrogram=show_spectrogram,
            show_summary=show_summary,
        )
    elif len(intensity_series) < max_len:
        # Show startup message while buffer is filling
        progress = len(intensity_series) / max_len
        msg = f"Data analysis starting up... {int(progress * 100)}%"
        cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    prev_gray = gray_small.copy()
    cv2.imshow("Wave Analyzer", frame)
    
    frame_time = max(1, int(1000 / fps))
    key = cv2.waitKey(frame_time)
    if key == ord('q'):
        break
    if key == ord('p'): # pause
        cv2.waitKey(-1) # any key release pause
    if key == ord('s'):
        show_spectrogram = not show_spectrogram
    if key == ord('a'):
        show_summary = not show_summary
    if key == ord('n'): # step frame by frame
        step = True
    if step:
        key = cv2.waitKey(-1) 
        if key != ord('n'): # use 'n' to step, any key other than 'n' releases step freeze
            step = False

cap.release()
cv2.destroyAllWindows()