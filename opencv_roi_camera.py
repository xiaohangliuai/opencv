"""Select a camera ROI with the mouse and show it in a separate live window."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2


CAMERA_WINDOW = "Camera - drag ROI | R: reset | Q/Esc: quit"
ROI_WINDOW = "ROI Camera"


@dataclass
class RoiSelector:
    """Store and update a rectangular selection from OpenCV mouse events."""

    start: tuple[int, int] | None = None
    end: tuple[int, int] | None = None
    roi: tuple[int, int, int, int] | None = None  # x1, y1, x2, y2
    dragging: bool = False

    def handle_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param

        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = (x, y)
            self.end = (x, y)
            self.dragging = True

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.end = (x, y)
            self.dragging = False
            self._finish_selection()

    def _finish_selection(self) -> None:
        if self.start is None or self.end is None:
            return

        x1, x2 = sorted((self.start[0], self.end[0]))
        y1, y2 = sorted((self.start[1], self.end[1]))

        # Ignore accidental clicks and selections too small to display.
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            self.roi = (x1, y1, x2, y2)

    def reset(self) -> None:
        self.start = None
        self.end = None
        self.roi = None
        self.dragging = False


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Open a camera, preferring DirectShow on Windows."""
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_index)
    return camera


def run_roi_camera(camera_index: int = 0) -> int:
    """Run the camera UI and continuously display the selected live ROI."""
    camera = open_camera(camera_index)
    if not camera.isOpened():
        print(
            f"Could not open camera {camera_index}. Close other camera apps, "
            "check camera permissions, or try --camera 1."
        )
        return 1

    selector = RoiSelector()
    roi_window_open = False

    cv2.namedWindow(CAMERA_WINDOW)
    cv2.setMouseCallback(CAMERA_WINDOW, selector.handle_mouse)
    print("Drag the left mouse button to select an ROI. Press R to reset or Q to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("The camera stopped returning frames.")
                return 1

            height, width = frame.shape[:2]
            preview = frame.copy()

            if selector.dragging and selector.start and selector.end:
                cv2.rectangle(preview, selector.start, selector.end, (0, 255, 0), 2)

            if selector.roi is not None:
                x1, y1, x2, y2 = selector.roi
                # Keep the selection valid if the camera resolution changes.
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)

                if x2 > x1 and y2 > y1:
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    roi_frame = frame[y1:y2, x1:x2]
                    cv2.imshow(ROI_WINDOW, roi_frame)
                    roi_window_open = True

            cv2.putText(
                preview,
                "Drag mouse: select ROI | R: reset | Q: quit",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(CAMERA_WINDOW, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                selector.reset()
                if roi_window_open:
                    cv2.destroyWindow(ROI_WINDOW)
                    roi_window_open = False
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select and display a live camera ROI.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index to use (default: 0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(run_roi_camera(arguments.camera))
