"""Track the largest blue object in a live OpenCV camera feed."""

from __future__ import annotations

import argparse
from collections import deque

import cv2
import numpy as np


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


def make_blue_mask(frame: np.ndarray) -> np.ndarray:
    """Create a cleaned binary mask of blue pixels in a BGR frame."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, LOWER_BLUE, UPPER_BLUE)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_blue_center(
    mask: np.ndarray, min_area: float
) -> tuple[tuple[int, int], tuple[int, int, int, int], float] | None:
    """Return the center, bounding box, and area of the largest blue object."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < min_area:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    center = (
        int(moments["m10"] / moments["m00"]),
        int(moments["m01"] / moments["m00"]),
    )
    bounding_box = cv2.boundingRect(contour)
    return center, bounding_box, area


def draw_trail(
    frame: np.ndarray, positions: deque[tuple[int, int] | None]
) -> None:
    """Draw recent tracked positions, with newer segments shown thicker."""
    total = len(positions)
    for index in range(1, total):
        newer = positions[index - 1]
        older = positions[index]
        if newer is None or older is None:
            continue

        thickness = max(1, int(6 * (total - index) / total))
        cv2.line(frame, newer, older, (0, 0, 255), thickness, cv2.LINE_AA)


def run_blue_tracker(
    camera_index: int = 0,
    min_area: float = 500.0,
    trail_length: int = 64,
) -> int:
    """Track the largest blue object and display its recent movement path."""
    camera = open_camera(camera_index)
    if not camera.isOpened():
        print(
            f"Could not open camera {camera_index}. Close other camera apps, "
            "check camera permissions, or try --camera 1."
        )
        return 1

    positions: deque[tuple[int, int] | None] = deque(maxlen=trail_length)
    print("Blue tracking started. Press C to clear the trail or Q/Esc to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("The camera stopped returning frames.")
                return 1

            mask = make_blue_mask(frame)
            tracked_object = find_blue_center(mask, min_area)

            if tracked_object is None:
                positions.appendleft(None)
                status = "Blue object: not found"
            else:
                center, (x, y, width, height), area = tracked_object
                positions.appendleft(center)

                cv2.rectangle(
                    frame, (x, y), (x + width, y + height), (0, 255, 0), 2
                )
                cv2.circle(frame, center, 6, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"center={center} area={area:.0f}px",
                    (x, max(50, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                status = "Blue object: tracking"

            draw_trail(frame, positions)
            cv2.putText(
                frame,
                f"{status} | C: clear | Q/Esc: quit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Blue Object Tracker", frame)
            cv2.imshow("Blue Mask", mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                positions.clear()
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track a blue object with a camera.")
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
    parser.add_argument(
        "--trail-length",
        type=int,
        default=64,
        help="Maximum number of positions in the motion trail (default: 64).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(
        run_blue_tracker(
            camera_index=arguments.camera,
            min_area=arguments.min_area,
            trail_length=max(1, arguments.trail_length),
        )
    )
