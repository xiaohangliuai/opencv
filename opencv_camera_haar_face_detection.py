"""Detect faces in a live camera feed with an OpenCV Haar cascade."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect faces from a camera using a Haar cascade."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index to use (default: 0). Try 1 for another camera.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("captures"),
        help="Folder for snapshots (default: captures).",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.1,
        help="Image-scale reduction per detection pass (default: 1.1).",
    )
    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
        help="Higher values reduce false detections (default: 5).",
    )
    return parser.parse_args()


def open_camera(camera_index: int) -> cv2.VideoCapture:
    """Open a camera, preferring DirectShow on Windows."""
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_index)
    return camera


def load_face_detector() -> cv2.CascadeClassifier:
    """Load OpenCV's bundled frontal-face Haar cascade."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")
    return detector


def detect_faces(
    frame,
    detector: cv2.CascadeClassifier,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
):
    """Return an annotated frame and the detected face rectangles."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    result = frame.copy()
    for x, y, width, height in faces:
        cv2.rectangle(
            result,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            result,
            "Face",
            (x, max(25, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        result,
        f"Faces: {len(faces)} | S: save | Q/Esc: quit",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return result, faces


def run_face_detection(
    camera_index: int = 0,
    output: Path = Path("captures"),
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
) -> int:
    """Continuously detect faces from a camera until the user exits."""
    if scale_factor <= 1.0:
        print("--scale-factor must be greater than 1.0.")
        return 2
    if min_neighbors < 0:
        print("--min-neighbors must be zero or greater.")
        return 2

    try:
        detector = load_face_detector()
    except RuntimeError as error:
        print(error)
        return 1

    camera = open_camera(camera_index)
    if not camera.isOpened():
        print(
            f"Could not open camera {camera_index}. Close other camera apps, "
            "check camera permissions, or try --camera 1."
        )
        return 1

    output.mkdir(parents=True, exist_ok=True)
    print("Face detection started. Press S to save a frame or Q/Esc to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("The camera stopped returning frames.")
                return 1

            result, _ = detect_faces(
                frame,
                detector,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors,
            )
            cv2.imshow("Haar Face Detection", result)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                image_path = output / f"faces_{timestamp}.jpg"
                if cv2.imwrite(str(image_path), result):
                    print(f"Saved {image_path.resolve()}")
                else:
                    print(f"Could not save {image_path}")
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(
        run_face_detection(
            camera_index=arguments.camera,
            output=arguments.output,
            scale_factor=arguments.scale_factor,
            min_neighbors=arguments.min_neighbors,
        )
    )
