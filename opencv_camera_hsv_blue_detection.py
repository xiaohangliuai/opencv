"""Detect blue objects in a live OpenCV camera feed."""

from __future__ import annotations

import argparse

import cv2
import numpy as np


# HSV is more reliable than raw BGR values for color detection.
# OpenCV hue values range from 0 to 179; blue is roughly 100 to 140.
LOWER_BLUE = np.array([100, 100, 50], dtype=np.uint8)
UPPER_BLUE = np.array([140, 255, 255], dtype=np.uint8)


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Open a camera, preferring DirectShow on Windows."""
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_index)
    return camera


def detect_blue(
    frame: np.ndarray, min_area: float = 500.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return an annotated frame and a mask containing detected blue areas."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, LOWER_BLUE, UPPER_BLUE)

    # Remove isolated pixels and fill small gaps inside detected objects.
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    annotated = frame.copy()
    detected_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        cv2.rectangle(
            annotated, (x, y), (x + width, y + height), (0, 255, 0), 2
        )
        cv2.putText(
            annotated,
            f"Blue ({area:.0f}px)",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        detected_count += 1

    cv2.putText(
        annotated,
        f"Blue objects: {detected_count} | Q/Esc: quit",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated, mask


def run_blue_detection(camera_index: int = 0, min_area: float = 500.0) -> int:
    """Open the camera and continuously detect blue objects."""
    camera = open_camera(camera_index)
    if not camera.isOpened():
        print(
            f"Could not open camera {camera_index}. Close other camera apps, "
            "check camera permissions, or try --camera 1."
        )
        return 1

    print("Blue detection started. Press Q or Esc to quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("The camera stopped returning frames.")
                return 1

            result, mask = detect_blue(frame, min_area)
            cv2.imshow("Blue Detection", result)
            cv2.imshow("Blue Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect blue objects with a camera.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index to use (default: 0).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=500.0,
        help="Ignore blue regions smaller than this pixel area (default: 500).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(run_blue_detection(arguments.camera, arguments.min_area))
