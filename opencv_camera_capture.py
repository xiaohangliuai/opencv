"""Preview a laptop camera and optionally save snapshots with OpenCV."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a live laptop-camera preview.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index to use (default: 0). Try 1 if another camera opens.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("captures"),
        help="Folder for snapshots (default: captures).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # CAP_DSHOW usually opens built-in webcams faster on Windows.
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        print(
            f"Could not open camera {args.camera}. Close other camera apps, "
            "check camera permissions, or try --camera 1."
        )
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    window_name = "Laptop Camera - S: save photo | Q/Esc: quit"

    print("Camera opened. Press S to save a photo; press Q or Esc to quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("The camera stopped returning frames.")
                return 1

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                image_path = args.output / f"photo_{timestamp}.jpg"
                if cv2.imwrite(str(image_path), frame):
                    print(f"Saved {image_path.resolve()}")
                else:
                    print(f"Could not save {image_path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
