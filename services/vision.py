import asyncio
import base64
from collections import Counter
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO

from config import settings
from schemas.audit import BoundingBox, VisionTelemetry


class VisionService:
    """Production service managing YOLOv8 inference and visual annotation."""

    def __init__(
        self,
        model_path: str = settings.yolo_model_path,
        conf_thresh: float = settings.yolo_confidence_threshold,
        restricted_zone_polygon: Optional[List[List[float]]] = None,
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.restricted_zone_polygon: List[List[float]] = (
            restricted_zone_polygon
            if restricted_zone_polygon is not None
            else getattr(settings, "restricted_zone_polygon", [[0.10, 0.35], [0.55, 0.35], [0.55, 0.95], [0.10, 0.95]])
        )
        self._model: YOLO | None = None

    @property
    def model(self) -> YOLO:
        """Lazy loader ensuring model is loaded once in memory."""
        if self._model is None:
            self._model = YOLO(self.model_path)
        return self._model

    def _get_polygon_pixels(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Scales normalized [0.0, 1.0] polygon vertices to current frame pixel dimensions."""
        h, w = frame_shape[:2]
        pts = []
        for pt in self.restricted_zone_polygon:
            px = int(round(pt[0] * w)) if pt[0] <= 1.0 else int(round(pt[0]))
            py = int(round(pt[1] * h)) if pt[1] <= 1.0 else int(round(pt[1]))
            pts.append([px, py])
        return np.array(pts, dtype=np.int32)

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[BoundingBox],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        draw_zone: bool = True
    ) -> np.ndarray:
        """
        Draw high-visibility bounding boxes and filled label tags.
        Normal detections use bright green (BGR: (0, 255, 0)).
        Restricted zone intrusions use bright red (BGR: (0, 0, 255)).
        Optionally renders semi-transparent restricted hazard zone polygon.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw semi-transparent restricted hazard zone polygon ROI
        if draw_zone and len(self.restricted_zone_polygon) >= 3:
            polygon_pts = self._get_polygon_pixels((h, w))
            overlay = annotated.copy()
            # Semi-transparent red hazard overlay
            cv2.fillPoly(overlay, [polygon_pts], (0, 0, 220))
            cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0, annotated)
            # Distinct red hazard boundary line
            cv2.polylines(annotated, [polygon_pts], isClosed=True, color=(0, 0, 255), thickness=2)
            # Hazard zone title banner
            min_x = min(pt[0] for pt in polygon_pts)
            min_y = min(pt[1] for pt in polygon_pts)
            tag_pos = (max(min_x, 10), max(min_y - 8, 20))
            cv2.putText(
                annotated,
                "RESTRICTED ZONE (HAZARD)",
                tag_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                lineType=cv2.LINE_AA
            )

        for det in detections:
            x1, y1, x2, y2 = [int(round(c)) for c in det.box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Determine colors based on intrusion status
            is_intrusion = getattr(det, "is_intrusion", False)
            det_color = (0, 0, 255) if is_intrusion else color
            text_color = (255, 255, 255) if is_intrusion else (0, 0, 0)

            # 1. Bounding box (Bright Red (0, 0, 255) if intrusion, Green (0, 255, 0) otherwise)
            box_thick = 3 if is_intrusion else thickness
            cv2.rectangle(annotated, (x1, y1), (x2, y2), det_color, box_thick)

            # 2. Formatted label tag text: [INTRUSION!] label: confidence%
            conf_pct = int(round(det.confidence * 100))
            prefix = "INTRUSION! " if is_intrusion else ""
            label_text = f"{prefix}{det.class_name} {conf_pct}%"

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

            # Filled label tag rectangle
            cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), det_color, cv2.FILLED)

            # Contrasting text
            cv2.putText(
                annotated,
                label_text,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
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
        """Inference, restricted zone intrusion detection, and telemetry extraction."""
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
                        box=coords,
                        is_intrusion=False
                    )
                )
                class_counter[class_name] += 1

        # 3. Detect Restricted Zone Intrusions via cv2.pointPolygonTest
        intrusion_detected = False
        polygon_pts = self._get_polygon_pixels(frame.shape)

        if len(polygon_pts) >= 3:
            for det in detections:
                if det.class_name.lower() == "person":
                    x1, y1, x2, y2 = det.box
                    bottom_center = ((x1 + x2) / 2.0, float(y2))
                    # cv2.pointPolygonTest returns positive (>0) inside, 0 on edge, negative (<0) outside
                    is_inside = cv2.pointPolygonTest(polygon_pts, bottom_center, False) >= 0
                    if is_inside:
                        det.is_intrusion = True
                        intrusion_detected = True

        # 4. Optionally encode annotated visualization with bounding boxes and zone overlay
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
            intrusion_detected=intrusion_detected,
            restricted_zone=self.restricted_zone_polygon,
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
