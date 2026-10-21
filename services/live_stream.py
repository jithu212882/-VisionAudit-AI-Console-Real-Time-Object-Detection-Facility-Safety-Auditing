import asyncio
from datetime import datetime, timezone
import json
import logging
import time
from typing import Optional, Set
import cv2
from fastapi import WebSocket, WebSocketDisconnect

from config import settings
from schemas.audit import (
    LiveStreamMessage,
    StreamEventType,
    VisionTelemetry,
    FacilityAuditReport,
)
from services.vision import vision_service
from services.reasoning import reasoning_service

logger = logging.getLogger("visionaudit.services.live_stream")


class LiveStreamAuditor:
    """Production live camera stream auditor managing YOLOv8 inference and throttled LLM audits."""

    def __init__(
        self,
        websocket: WebSocket,
        source: str = "0",
        include_preview: bool = True,
        audit_interval: float = settings.live_audit_interval_seconds,
        critical_cooldown: float = settings.critical_alert_cooldown_seconds,
    ):
        self.websocket = websocket
        self.raw_source = source
        self.include_preview = include_preview
        self.audit_interval = audit_interval
        self.critical_cooldown = critical_cooldown
        self.critical_classes: Set[str] = set(settings.critical_object_classes)

        # State tracking
        self.is_running = False
        self.last_periodic_audit_time = 0.0
        self.last_critical_audit_time = 0.0
        self.active_audit_task: Optional[asyncio.Task] = None
        self.prev_critical_counts: dict[str, int] = {}
        
        # Exponential backoff governor
        self.consecutive_failures = 0
        self.backoff_until = 0.0

    def _parse_source(self, src: str):
        """Converts numerical string index (e.g. '0') to integer, or returns RTSP/URL string."""
        stripped = src.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped

    async def _dispatch_llm_audit(
        self,
        telemetry: VisionTelemetry,
        trigger_reason: str,
        event_type: StreamEventType,
    ):
        """Asynchronously triggers the LangChain LCEL audit without stalling camera frame rate."""
        current_ts = datetime.now(timezone.utc).isoformat()
        try:
            logger.info(f"Triggering throttled LLM audit: {trigger_reason}")
            report: FacilityAuditReport = await reasoning_service.audit_telemetry(
                telemetry=telemetry,
                timestamp=current_ts
            )

            # Reset consecutive failures on success
            self.consecutive_failures = 0
            self.backoff_until = 0.0

            # Send the audit report over WebSocket
            alert_msg = LiveStreamMessage(
                event=event_type,
                timestamp=current_ts,
                report=report,
                trigger_reason=trigger_reason,
                telemetry=telemetry
            )
            await self.websocket.send_text(alert_msg.model_dump_json())

        except Exception as exc:
            self.consecutive_failures += 1
            err_str = str(exc)
            
            # Identify 429 rate limit / quota exhaustion
            is_rate_limit = (
                "429" in err_str 
                or "quota" in err_str.lower() 
                or "resourceexhausted" in err_str.lower()
            )
            
            # Exponential backoff calculation (30s base for 429, 15s for generic errors, max 300s)
            base_pause = 30.0 if is_rate_limit else 15.0
            pause_sec = min(base_pause * (2 ** min(self.consecutive_failures - 1, 4)), 300.0)
            
            self.backoff_until = time.time() + pause_sec
            self.last_periodic_audit_time = self.backoff_until
            self.last_critical_audit_time = self.backoff_until

            logger.error(
                f"LLM audit failed ({self.consecutive_failures}x). Backing off for {int(pause_sec)}s. Error: {exc}",
                exc_info=True
            )
            try:
                error_msg = LiveStreamMessage(
                    event=StreamEventType.ERROR,
                    timestamp=current_ts,
                    message=(
                        f"LLM reasoning paused for {int(pause_sec)}s "
                        f"({'Quota Exceeded / Rate Limit' if is_rate_limit else 'Service Error'}): {err_str[:120]}"
                    )
                )
                await self.websocket.send_text(error_msg.model_dump_json())
            except Exception:
                pass
        finally:
            self.active_audit_task = None

    def _should_trigger_critical(self, telemetry: VisionTelemetry, now: float) -> tuple[bool, str]:
        """Evaluates if new critical objects (e.g., person, fire, weapons) appeared or changed."""
        if (now - self.last_critical_audit_time) < self.critical_cooldown:
            return False, ""

        detected_critical = []
        is_triggered = False

        for cls_name, count in telemetry.class_counts.items():
            if cls_name.lower() in self.critical_classes:
                prev_count = self.prev_critical_counts.get(cls_name.lower(), 0)
                # Trigger if newly appeared or increased count
                if count > prev_count or (now - self.last_critical_audit_time >= self.audit_interval):
                    is_triggered = True
                    detected_critical.append(f"{cls_name} ({count})")

        if is_triggered:
            reason = f"Critical entities identified: {', '.join(detected_critical)}"
            return True, reason

        return False, ""

    async def run(self):
        """Main camera acquisition and continuous YOLO processing loop."""
        parsed_source = self._parse_source(self.raw_source)
        logger.info(f"Opening video capture source: {parsed_source}")

        # Open video capture asynchronously
        cap = await asyncio.to_thread(cv2.VideoCapture, parsed_source)
        
        if not cap.isOpened():
            err_msg = LiveStreamMessage(
                event=StreamEventType.ERROR,
                timestamp=datetime.now(timezone.utc).isoformat(),
                message=f"Failed to open video source: '{self.raw_source}'. Verify camera availability or RTSP URL."
            )
            await self.websocket.send_text(err_msg.model_dump_json())
            return

        # Notify client of successful connection
        connected_msg = LiveStreamMessage(
            event=StreamEventType.STATUS,
            timestamp=datetime.now(timezone.utc).isoformat(),
            message=f"Camera feed connected to source: {self.raw_source}. Continuous YOLOv8 auditing active."
        )
        await self.websocket.send_text(connected_msg.model_dump_json())

        self.is_running = True
        self.last_periodic_audit_time = time.time()
        self.last_critical_audit_time = 0.0

        prev_time = time.time()
        fps = 0.0

        try:
            while self.is_running:
                # 1. Grab frame in thread pool
                ret, frame = await asyncio.to_thread(cap.read)
                if not ret or frame is None:
                    logger.warning("Frame read returned False or empty. Stream may have terminated.")
                    await asyncio.sleep(0.1)
                    continue

                # 2. Calculate FPS
                now = time.time()
                elapsed = now - prev_time
                if elapsed > 0:
                    fps = round(1.0 / elapsed, 1)
                prev_time = now

                # 3. Continuous YOLOv8 detection
                telemetry: VisionTelemetry = await vision_service.analyze_cv2_frame(
                    frame=frame,
                    include_annotated_base64=self.include_preview,
                    include_preview=self.include_preview
                )

                # 4. Stream real-time telemetry back to client
                current_iso = datetime.now(timezone.utc).isoformat()
                telemetry_msg = LiveStreamMessage(
                    event=StreamEventType.TELEMETRY,
                    timestamp=current_iso,
                    fps=fps,
                    telemetry=telemetry,
                    preview=telemetry.annotated_image_base64 if self.include_preview else None
                )
                await self.websocket.send_text(telemetry_msg.model_dump_json())

                # 5. Check LangChain Throttling Rules:
                # Respect active backoff pause and check if no audit is in-flight
                if now >= self.backoff_until and (self.active_audit_task is None or self.active_audit_task.done()):
                    # Rule A: Critical Object Trigger
                    should_crit, crit_reason = self._should_trigger_critical(telemetry, now)
                    if should_crit:
                        self.last_critical_audit_time = now
                        self.last_periodic_audit_time = now  # Reset periodic timer
                        self.active_audit_task = asyncio.create_task(
                            self._dispatch_llm_audit(
                                telemetry=telemetry,
                                trigger_reason=crit_reason,
                                event_type=StreamEventType.CRITICAL_ALERT
                            )
                        )
                    # Rule B: Periodic Interval Trigger (e.g. every 10 seconds summary)
                    elif (now - self.last_periodic_audit_time) >= self.audit_interval:
                        self.last_periodic_audit_time = now
                        reason = f"Periodic {int(self.audit_interval)}-second facility audit summary"
                        self.active_audit_task = asyncio.create_task(
                            self._dispatch_llm_audit(
                                telemetry=telemetry,
                                trigger_reason=reason,
                                event_type=StreamEventType.PERIODIC_AUDIT
                            )
                        )

                # Update critical count memory
                self.prev_critical_counts = {
                    cls_name.lower(): count
                    for cls_name, count in telemetry.class_counts.items()
                    if cls_name.lower() in self.critical_classes
                }

                # Yield control back to event loop
                await asyncio.sleep(0.001)

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected from live audit stream.")
        except asyncio.CancelledError:
            logger.info("Live audit task cancelled.")
        except Exception as exc:
            logger.error(f"Live audit stream encountered an error: {exc}", exc_info=True)
            try:
                err_msg = LiveStreamMessage(
                    event=StreamEventType.ERROR,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    message=f"Live stream interrupted: {str(exc)}"
                )
                await self.websocket.send_text(err_msg.model_dump_json())
            except Exception:
                pass
        finally:
            self.is_running = False
            if cap:
                await asyncio.to_thread(cap.release)
                logger.info(f"Released video capture source '{self.raw_source}'.")
            if self.active_audit_task and not self.active_audit_task.done():
                self.active_audit_task.cancel()
