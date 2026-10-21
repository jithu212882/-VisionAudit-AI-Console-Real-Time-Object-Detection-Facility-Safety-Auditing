from datetime import datetime, timezone
import logging
from typing import Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, status, WebSocket, Query

from config import settings
from services.vision import vision_service
from services.reasoning import reasoning_service
from services.live_stream import LiveStreamAuditor
from schemas.audit import VisionAuditResponse, VisionTelemetry, FacilityAuditReport

logger = logging.getLogger("visionaudit.routers.audit")

router = APIRouter(prefix="/audit", tags=["Facility Audit"])

# Max upload limit: 20 MB
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "application/octet-stream"
}


@router.post(
    "/analyze-frame",
    response_model=VisionAuditResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze image frame for visual telemetry and policy audit report",
    description=(
        "Accepts a multipart image upload, executes YOLOv8 object detection, extracts "
        "bounding boxes & class frequencies, and invokes Gemini 1.5-flash via LangChain LCEL "
        "to output a structured facility compliance and risk audit report."
    ),
    responses={
        200: {"description": "Successful audit and visual telemetry extraction."},
        400: {"description": "Corrupted or invalid image upload."},
        413: {"description": "Uploaded image file exceeds maximum allowed size (20MB)."},
        500: {"description": "Inference failure or server configuration error."},
        502: {"description": "Downstream LLM reasoning service failure."},
    }
)
async def analyze_frame(
    file: UploadFile = File(..., description="Image or video frame file (JPEG, PNG, WebP)"),
    timestamp: Optional[str] = Form(
        None, 
        description="Optional ISO timestamp of frame capture. Defaults to server UTC now."
    )
) -> VisionAuditResponse:
    """End-to-end pipeline processing a camera frame through Vision and LLM layers."""
    
    # 1. Validate file presence
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file was provided in the upload request."
        )

    # 2. Validate MIME content type
    if file.content_type and file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported media type '{file.content_type}'. Must be JPEG, PNG, BMP, or WebP."
        )

    # 3. Read image byte payload safely with size check
    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes)."
            )
        if len(image_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image file exceeds maximum limit of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB."
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to read incoming file stream: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file payload."
        )

    # 4. Vision Layer: Execute YOLOv8 detection and annotation
    try:
        telemetry: VisionTelemetry = await vision_service.analyze_frame(image_bytes)
    except ValueError as val_err:
        logger.warning(f"Invalid image format: {val_err}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as cv_err:
        logger.error(f"Computer Vision processing failed: {cv_err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Computer Vision inference pipeline error: {str(cv_err)}"
        )

    # 5. Determine audit timestamp
    effective_timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    # 6. Reasoning Layer: Execute LangChain LCEL + Gemini 1.5-flash audit
    try:
        report: FacilityAuditReport = await reasoning_service.audit_telemetry(
            telemetry=telemetry,
            timestamp=effective_timestamp
        )
    except ValueError as cfg_err:
        logger.error(f"Reasoning service configuration error: {cfg_err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(cfg_err)
        )
    except Exception as llm_err:
        logger.error(f"Downstream LLM audit failure: {llm_err}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM Reasoning Service error during audit synthesis: {str(llm_err)}"
        )

    return VisionAuditResponse(
        status="success",
        telemetry=telemetry,
        report=report
    )


@router.websocket("/ws/live-audit")
async def live_audit_stream(
    websocket: WebSocket,
    source: str = Query("0", description="Webcam index (e.g. '0') or RTSP stream URL"),
    preview: bool = Query(True, description="Include base64 annotated frame in telemetry"),
    interval: float = Query(
        settings.live_audit_interval_seconds,
        description="Periodic LLM audit interval in seconds"
    ),
    cooldown: float = Query(
        settings.critical_alert_cooldown_seconds,
        description="Cooldown in seconds between critical alerts"
    )
):
    """
    Live WebSocket streaming audit endpoint.
    Continuously processes frames with YOLOv8 and throttles Gemini 1.5-flash
    reasoning using critical entity triggers and periodic interval summaries.
    """
    await websocket.accept()
    auditor = LiveStreamAuditor(
        websocket=websocket,
        source=source,
        include_preview=preview,
        audit_interval=interval,
        critical_cooldown=cooldown
    )
    await auditor.run()

