# Wave Video Water

This project processes live camera or video input for wave analysis, overlays, and OSC-driven control.

## Python Virtualenv

The project uses a local Python 3.10 virtual environment in `.venv`.

Create it and install dependencies:

```powershell
cd "C:\Cabbage_VST\CabbageEfx\wave_video_water"
& "C:\Users\obran\AppData\Local\Programs\Python\Python310\python.exe" -m venv ".venv"
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r "requirements.txt" pyusb
```

Activate it in PowerShell:

```powershell
cd "C:\Cabbage_VST\CabbageEfx\wave_video_water"
.\.venv\Scripts\Activate.ps1
```

If activation is blocked, run scripts with the venv Python directly:

```powershell
& ".venv\Scripts\python.exe" .\main.py
```

## Calibration

Use the standalone Blackmagic calibration helper to prepare the camera before analysis:

```powershell
& ".venv\Scripts\python.exe" .\blackmagic_calibration.py
```

If you already know the ATEM address and camera input, you can pass them directly:

```powershell
& ".venv\Scripts\python.exe" .\blackmagic_calibration.py --ip 172.31.57.153 --camera 1
```

The calibrator will also prompt for an ATEM IP and camera/input number if you omit them.

## Run The Main Program

Start the wave analyzer with:

```powershell
& ".venv\Scripts\python.exe" .\main.py
```

On startup, choose either a live camera or a video file. For live camera input, the app scans available camera indices and lets you pick one.

## Overlay And Panel Keys

The main window uses keyboard shortcuts to toggle overlays and diagnostics:

- `v` cycles flow display: off -> arrows -> full.
- `x` toggles axial flow mode.
- `w` toggles the texture overlay.
- `c` toggles the contour overlay.
- `r` toggles the threshold overlay.
- `s` toggles the spectrum overlay.
- `y` toggles the CPU profile panel.
- `m` toggles the mask overlay.
- `j` toggles LBP analysis/overlay.
- `h` toggles temporal-difference mode.
- `g` cycles temporal-difference polarity.
- `e` cycles screen-blend mode.
- `b` cycles blur mode.
- `f` cycles flow-detail mode.

Useful non-overlay controls:

- `q` quit.
- `p` pause.
- `n` step one frame while paused.
- `d` cycle display mode.
- `t` cycle temporal filter mode.
- `i` toggle automatic flow update interval.
- `k` enter mask edit mode.
- `l` load mask.
- `L` save mask.
- `1` to `4` select quality presets.

## Notes

- Live camera capture is downsampled early so the analyzer does not spend CPU on oversized frames.
- The calibration script keeps camera-side settings in place after it exits, assuming the camera remains powered.