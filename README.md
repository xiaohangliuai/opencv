# OpenCV Camera Examples

Small Python examples for working with a webcam using OpenCV:

- Preview the camera and save snapshots.
- Detect blue objects in a live camera feed.
- Select and display a live region of interest (ROI).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Usage

```powershell
python opencv_camera_capture.py
python opencv_camera_hsv_blue_detection.py
python opencv_roi_camera.py
```

Pass `--camera 1` if your default camera is not at index `0`. Run any script
with `--help` to see its available options.

