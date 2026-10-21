"""
Vision Pipeline & Desktop Visualizer Unit Tests.
Verifies YOLOv8 model loading, green bounding box annotations, contrasting text tags, and desktop HUD rendering.
"""

import sys
from pathlib import Path
import base64

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from schemas.audit import BoundingBox, VisionTelemetry
from services.vision import vision_service
import test_desktop_view


def test_yolo_model_lazy_loader():
    model = vision_service.model
    assert model is not None, "YOLOv8 model failed to initialize."
    assert hasattr(model, "predict"), "YOLO model missing predict method."


def test_green_bounding_box_annotation():
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        BoundingBox(class_id=0, class_name="person", confidence=0.95, box=[50.0, 50.0, 200.0, 300.0]),
        BoundingBox(class_id=63, class_name="laptop", confidence=0.88, box=[300.0, 200.0, 500.0, 400.0])
    ]

    annotated = vision_service.annotate_frame(canvas, detections)
    assert annotated.shape == (480, 640, 3)

    # Validate green bounding box edge at (x=50, y=50)
    assert list(annotated[50, 50]) == [0, 255, 0], f"Pixel at (50,50) should be BGR [0, 255, 0], got {annotated[50, 50]}"

    # Validate filled tag background above box (at y=30, x=52)
    assert list(annotated[30, 52]) == [0, 255, 0], f"Filled tag background should be BGR [0, 255, 0], got {annotated[30, 52]}"


def test_preview_base64_generation():
    canvas = np.zeros((320, 320, 3), dtype=np.uint8)
    telemetry: VisionTelemetry = vision_service._process_frame_matrix(canvas, include_annotated_base64=True)
    assert telemetry.annotated_image_base64 is not None, "Base64 preview was not populated."

    raw_bytes = base64.b64decode(telemetry.annotated_image_base64)
    np_buf = np.frombuffer(raw_bytes, np.uint8)
    decoded_img = cv2.imdecode(np_buf, cv2.IMREAD_COLOR)
    assert decoded_img is not None
    assert decoded_img.shape == (320, 320, 3)


def test_desktop_view_hud_rendering():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        BoundingBox(class_id=0, class_name="person", confidence=0.92, box=[50.0, 50.0, 200.0, 300.0])
    ]
    hud_frame = test_desktop_view.draw_hud(frame, fps=30.0, detections=detections)
    assert hud_frame is not None
    assert hud_frame.shape == (480, 640, 3)


if __name__ == "__main__":
    print("[*] Running YOLO Vision Pipeline & Desktop View Tests...")
    test_yolo_model_lazy_loader()
    test_green_bounding_box_annotation()
    test_preview_base64_generation()
    test_desktop_view_hud_rendering()
    print("[+] All Vision Pipeline tests passed successfully!")
