import cv2
import os
import numpy as np
from pathlib import Path
from video_capture import get_frame
from wave_analysis import analyze_direction, analyze_spatial_frequencies
from spectrum_plot import render_spectrum_overlay


def select_video_source():
    """
    Present a menu to select video source: live camera or a video file from ../wave_video_files/
    Returns the video source (0 for camera, or file path for video)
    """
    video_dir = Path("../wave_video_files/")
    
    # Get list of video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    video_files = sorted([
        f for f in os.listdir(video_dir) 
        if os.path.isfile(video_dir / f) and Path(f).suffix.lower() in video_extensions
    ])
    
    print("\n" + "="*50)
    print("VIDEO SOURCE SELECTION")
    print("="*50)
    print("\n0. Live Camera (Webcam)")
    
    for i, video_file in enumerate(video_files, start=1):
        print(f"{i}. {video_file}")
    
    print("\n" + "-"*50)
    while True:
        try:
            choice = input("Select video source (0 for camera, or enter number): ").strip()
            choice_num = int(choice)
            
            if choice_num == 0:
                print("\nUsing live camera (index 0)...\n")
                return 0
            elif 1 <= choice_num <= len(video_files):
                selected_file = video_files[choice_num - 1]
                file_path = str(video_dir / selected_file)
                print(f"\nUsing video file: {selected_file}\n")
                return file_path
            else:
                print(f"Invalid choice. Please enter 0-{len(video_files)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


# Get video source from user selection
video_source = select_video_source()
cap = cv2.VideoCapture(video_source)
step = False
show_spectrogram = True
show_summary = True

fps = cap.get(cv2.CAP_PROP_FPS) or 30
print(f"Video FPS: {fps:.2f}")
prev_gray = None

while True:
    result = get_frame(cap)
    if result is None:
        break
    frame, gray = result
    gray_small = cv2.resize(gray, (160, 120))

    if prev_gray is not None:
        direction = analyze_direction(prev_gray, gray_small)
        spatial_freqs = analyze_spatial_frequencies(gray)
        render_spectrum_overlay(
            frame,
            np.array([]),  # xf (empty, no temporal freq)
            np.array([]),  # yf (empty, no temporal freq)
            {},            # peaks (empty, no temporal freq)
            spatial_freqs=spatial_freqs,
            show_spectrogram=True,  # Show spatial spectrum
            show_summary=show_summary,
        )

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