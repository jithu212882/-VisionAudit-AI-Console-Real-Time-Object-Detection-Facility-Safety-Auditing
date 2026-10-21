"""
VisionAudit - Standalone Desktop Live View
Opens a local OpenCV window (cv2.imshow) displaying the live camera feed
with real-time high-visibility green bounding boxes, detection tags, and a live HUD.

Press 'q' or ESC to exit.
"""

import argparse
import sys
import time
from collections import Counter
from typing import List
import cv2
import numpy as np

from config import settings
from schemas.audit import BoundingBox
from services.vision import vision_service


def parse_source(src: str):
    """Parse string input into an integer camera index or string stream/file path."""
    stripped = src.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def draw_hud(frame: np.ndarray, fps: float, detections: List[BoundingBox]) -> np.ndarray:
    """Draw a modern, non-intrusive status HUD bar at the top of the OpenCV window."""
    h, w = frame.shape[:2]
    bar_height = 42

    # Semi-transparent dark banner background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (15, 23, 42), -1)
    # Bottom border line of banner (cyan/green accent)
    cv2.line(overlay, (0, bar_height), (w, bar_height), (0, 255, 0), 1)
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Green status beacon (circle)
    cv2.circle(frame, (18, 21), 6, (0, 255, 0), -1)

    # Title & FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
        "VisionAudit Live Feed",
        (34, 26),
        font,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA
    )

    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (220, 26),
        font,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

    # Detection count & classes
    counts = Counter([d.class_name for d in detections])
    summary_str = f"Objects: {len(detections)}"
    if counts:
        classes_str = ", ".join([f"{k}:{v}" for k, v in list(counts.items())[:3]])
        summary_str += f" ({classes_str})"

    cv2.putText(
        frame,
        summary_str,
        (330, 26),
        font,
        0.5,
        (56, 189, 248),  # light blue
        1,
        cv2.LINE_AA
    )

    # Exit hint on the right
    exit_text = "Press 'q' to exit"
    (tw, _), _ = cv2.getTextSize(exit_text, font, 0.45, 1)
    cv2.putText(
        frame,
        exit_text,
        (max(10, w - tw - 16), 26),
        font,
        0.45,
        (148, 163, 184),
        1,
        cv2.LINE_AA
    )

    return frame


def run_desktop_stream(source: str = "0", conf_thresh: float = settings.yolo_confidence_threshold):
    """Main desktop capture, inference, annotation, and display loop."""
    parsed_source = parse_source(source)
    print("=" * 65)
    print("  VisionAudit Desktop Live View")
    print(f"  Target Source: '{source}' | Confidence: {conf_thresh:.2f}")
    print("  Controls: Press 'q' or ESC in the window to quit")
    print("=" * 65)

    cap = cv2.VideoCapture(parsed_source)
    is_live_capture = True

    if not cap.isOpened():
        print(f"\n[-] WARNING: Could not open video source '{source}'.")
        print("    If testing on a machine without a webcam, verify the camera index or file path.")
        print("    Attempting fallback to test image or synthetic camera stream...")
        cap.release()
        is_live_capture = False

    window_name = "VisionAudit - Live Feed (Press 'q' to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    # Model warm-up
    _ = vision_service.model

    prev_time = time.time()
    fps = 0.0
    frame_count = 0
    total_detections_count = 0

    try:
        if is_live_capture:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # In case of video file end, loop back or exit
                    if isinstance(parsed_source, str) and not parsed_source.isdigit():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print("\n[*] Camera feed ended or unavailable.")
                    break

                # FPS calculation
                now = time.time()
                elapsed = now - prev_time
                if elapsed > 0:
                    fps = 1.0 / elapsed
                prev_time = now

                # Run YOLO inference
                results = vision_service.model.predict(
                    source=frame,
                    conf=conf_thresh,
                    verbose=False
                )[0]

                detections: List[BoundingBox] = []
                if results.boxes is not None and len(results.boxes) > 0:
                    for box in results.boxes:
                        coords = [round(float(c), 2) for c in box.xyxy[0].tolist()]
                        conf = round(float(box.conf[0]), 4)
                        class_id = int(box.cls[0])
                        class_name = vision_service.model.names.get(class_id, f"unknown_{class_id}")
                        detections.append(
                            BoundingBox(
                                class_id=class_id,
                                class_name=class_name,
                                confidence=conf,
                                box=coords
                            )
                        )

                total_detections_count += len(detections)
                frame_count += 1

                # 1. Draw high-visibility green bounding boxes & label tags
                annotated = vision_service.annotate_frame(frame, detections)

                # 2. Draw HUD banner
                annotated = draw_hud(annotated, fps, detections)

                # 3. Render window
                cv2.imshow(window_name, annotated)

                # Check for exit key ('q' or ESC) or window close
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    print("\n[*] 'q' pressed. Closing desktop stream...")
                    break

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\n[*] Window closed by user.")
                    break

        else:
            # Fallback synthetic demo mode for headless / non-webcam verification
            print("[+] Running interactive synthetic test feed (press 'q' to exit)...")
            synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Create a nice dark gradient background
            for row in range(480):
                synthetic_frame[row, :] = [30 + int(row * 0.05), 20 + int(row * 0.04), 15 + int(row * 0.02)]

            demo_detections = [
                BoundingBox(class_id=0, class_name="person", confidence=0.94, box=[80.0, 100.0, 260.0, 420.0]),
                BoundingBox(class_id=63, class_name="laptop", confidence=0.88, box=[320.0, 220.0, 560.0, 400.0]),
            ]

            while True:
                now = time.time()
                elapsed = now - prev_time
                fps = 1.0 / max(0.001, elapsed)
                prev_time = now

                annotated = vision_service.annotate_frame(synthetic_frame, demo_detections)
                annotated = draw_hud(annotated, fps, demo_detections)
                cv2.imshow(window_name, annotated)

                key = cv2.waitKey(30) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    print("\n[*] 'q' pressed. Closing synthetic demo feed...")
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

    except KeyboardInterrupt:
        print("\n[*] Desktop live stream interrupted by user.")
    except Exception as exc:
        print(f"\n[-] Desktop view error: {exc}")
    finally:
        if is_live_capture and cap:
            cap.release()
        cv2.destroyAllWindows()
        print("=" * 65)
        print("  Stream Summary:")
        print(f"  Processed Frames: {frame_count}")
        print(f"  Detections Made: {total_detections_count}")
        print("  OpenCV windows destroyed successfully.")
        print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionAudit Native Desktop OpenCV Live View")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Webcam index (e.g. '0', '1') or video file path / RTSP URL"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=settings.yolo_confidence_threshold,
        help="Confidence threshold for YOLO object detection"
    )
    args = parser.parse_args()

    run_desktop_stream(source=args.source, conf_thresh=args.conf)
