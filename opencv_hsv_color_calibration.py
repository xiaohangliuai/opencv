"""Tune lower and upper HSV color thresholds with six live trackbars."""

from __future__ import annotations

import argparse

import cv2
import numpy as np


CONTROL_WINDOW = "HSV Controls"
CAMERA_WINDOW = "Camera"
MASK_WINDOW = "HSV Mask"
RESULT_WINDOW = "Detected Color"


def nothing(value: int) -> None:
    """OpenCV trackbars require a callback, even when no action is needed."""
    del value


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Open a camera, preferring DirectShow on Windows."""
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_index)
    return camera


def create_hsv_controls() -> None:
    """Create the six HSV lower/upper-bound trackbars."""
    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW, 500, 320)

    # Hue uses OpenCV's 0-179 range. Saturation and Value use 0-255.
    cv2.createTrackbar("Lower H", CONTROL_WINDOW, 0, 179, nothing)
    cv2.createTrackbar("Lower S", CONTROL_WINDOW, 0, 255, nothing)
    cv2.createTrackbar("Lower V", CONTROL_WINDOW, 0, 255, nothing)
    cv2.createTrackbar("Upper H", CONTROL_WINDOW, 179, 179, nothing)
    cv2.createTrackbar("Upper S", CONTROL_WINDOW, 255, 255, nothing)
    cv2.createTrackbar("Upper V", CONTROL_WINDOW, 255, 255, nothing)


def read_hsv_controls() -> tuple[np.ndarray, np.ndarray]:
    """Read the trackbars and return lower and upper HSV arrays."""
    lower = np.array(
        [
            cv2.getTrackbarPos("Lower H", CONTROL_WINDOW),
            cv2.getTrackbarPos("Lower S", CONTROL_WINDOW),
            cv2.getTrackbarPos("Lower V", CONTROL_WINDOW),
        ],
        dtype=np.uint8,
    )
    upper = np.array(
        [
            cv2.getTrackbarPos("Upper H", CONTROL_WINDOW),
            cv2.getTrackbarPos("Upper S", CONTROL_WINDOW),
            cv2.getTrackbarPos("Upper V", CONTROL_WINDOW),
        ],
        dtype=np.uint8,
    )
    return lower, upper


def format_thresholds(lower: np.ndarray, upper: np.ndarray) -> tuple[str, str]:
    """Format thresholds as copyable Python source code."""
    lower_text = f"LOWER_HSV = np.array({lower.tolist()}, dtype=np.uint8)"
    upper_text = f"UPPER_HSV = np.array({upper.tolist()}, dtype=np.uint8)"
    return lower_text, upper_text


def run_hsv_color_detector(camera_index: int = 0) -> int:
    """Run a camera preview with interactive HSV threshold controls."""
    camera = open_camera(camera_index)
    if not camera.isOpened():
        print(
            f"Could not open camera {camera_index}. Close other camera apps, "
            "check camera permissions, or try --camera 1."
        )
        return 1

    create_hsv_controls()
    print("Adjust the six HSV sliders. Press P to print values or Q/Esc to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("The camera stopped returning frames.")
                return 1

            lower, upper = read_hsv_controls()
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_frame, lower, upper)
            detected = cv2.bitwise_and(frame, frame, mask=mask)

            lower_label = f"Lower HSV: {lower.tolist()}"
            upper_label = f"Upper HSV: {upper.tolist()}"
            cv2.putText(
                frame,
                lower_label,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                upper_label,
                (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "P: print values | Q/Esc: quit",
                (10, 79),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(CAMERA_WINDOW, frame)
            cv2.imshow(MASK_WINDOW, mask)
            cv2.imshow(RESULT_WINDOW, detected)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                lower_text, upper_text = format_thresholds(lower, upper)
                print("\nCopy these values into your detector:")
                print(lower_text)
                print(upper_text)
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune lower and upper HSV color thresholds using six sliders."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index to use (default: 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(run_hsv_color_detector(arguments.camera))
