import asyncio
import base64
from collections import Counter
from typing import List, Dict, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from config import settings
from schemas.audit import BoundingBox, VisionTelemetry


class VisionService:
    """Production service managing YOLOv8 inference and visual annotation."""

    def __init__(self, model_path: str = settings.yolo_model_path, conf_thresh: float = settings.yolo_confidence_threshold):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self._model: YOLO | None = None

    @property
    def model(self) -> YOLO:
        """Lazy loader ensuring model is loaded once in memory."""
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[BoundingBox],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw high-visibility green bounding boxes (BGR: (0, 255, 0), thickness: 2)
        and filled label tags displaying 'label: confidence%' (e.g., person 94%, laptop 88%)
        with clear contrasting text above detected objects.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = [int(round(c)) for c in det.box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # 1. High-visibility green bounding box (BGR: (0, 255, 0), thickness 2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # 2. Formatted label tag text: label: confidence% (e.g. person 94%, laptop 88%)
            conf_pct = int(round(det.confidence * 100))
            label_text = f"{det.class_name} {conf_pct}%"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

            padding = 4
            tag_w = text_w + (padding * 2)
            tag_h = text_h + baseline + (padding * 2)

            # Position tag above bounding box; if too close to frame top, place inside top edge
            if y1 - tag_h >= 0:
                tag_y1 = y1 - tag_h
                tag_y2 = y1
                text_y = y1 - baseline - padding + 1
            else:
                tag_y1 = y1
                tag_y2 = y1 + tag_h
                text_y = y1 + text_h + padding

            tag_x1 = x1
            if tag_x1 + tag_w > w:
                tag_x1 = max(0, w - tag_w)
            tag_x2 = min(w, tag_x1 + tag_w)
            text_x = tag_x1 + padding

            # Green filled label tag rectangle
            cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), color, cv2.FILLED)

            # Contrasting text: black text (0, 0, 0) provides maximum legibility over bright green (0, 255, 0)
            cv2.putText(
                annotated,
                label_text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness,
                lineType=cv2.LINE_AA
            )

        return annotated

    def _process_frame_matrix(
        self,
        frame: np.ndarray,
        include_annotated_base64: bool = True,
        include_preview: bool | None = None
    ) -> VisionTelemetry:
        """Inference and telemetry extraction on an in-memory BGR numpy matrix."""
        if include_preview is not None:
            include_annotated_base64 = include_preview

        # 1. Run YOLOv8 inference with strict confidence threshold
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            verbose=False
        )[0]

        # 2. Extract telemetry: bounding boxes, confidence, class names
        detections: List[BoundingBox] = []
        class_counter: Counter = Counter()

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                coords = [round(float(c), 2) for c in box.xyxy[0].tolist()]
                confidence = round(float(box.conf[0]), 4)
                class_id = int(box.cls[0])
                class_name = self.model.names.get(class_id, f"unknown_{class_id}")

                detections.append(
                    BoundingBox(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        box=coords
                    )
                )
                class_counter[class_name] += 1

        # 3. Optionally encode annotated visualization with green bounding boxes and label tags
        annotated_b64: str | None = None
        if include_annotated_base64:
            annotated_frame = self.annotate_frame(frame, detections)
            success, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if success:
                annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        return VisionTelemetry(
            total_count=len(detections),
            class_counts=dict(class_counter),
            detections=detections,
            annotated_image_base64=annotated_b64
        )

    def _sync_process_frame(self, image_bytes: bytes) -> VisionTelemetry:
        """Synchronous CPU/GPU bound image decoding and inference from raw bytes."""
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Corrupted or unsupported image file. Failed to decode image bytes.")

        return self._process_frame_matrix(frame, include_annotated_base64=True)

    async def analyze_frame(self, image_bytes: bytes) -> VisionTelemetry:
        """Asynchronously process frame from byte payload."""
        return await asyncio.to_thread(self._sync_process_frame, image_bytes)

    async def analyze_cv2_frame(
        self,
        frame: np.ndarray,
        include_annotated_base64: bool = False,
        include_preview: bool | None = None
    ) -> VisionTelemetry:
        """Asynchronously process continuous camera frame (BGR numpy array) without encode/decode overhead."""
        if include_preview is not None:
            include_annotated_base64 = include_preview
        return await asyncio.to_thread(self._process_frame_matrix, frame, include_annotated_base64)


# Singleton instance
vision_service = VisionService()
