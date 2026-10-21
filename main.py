from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, Request, status, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from config import settings
from routers.audit import router as audit_router
from services.vision import vision_service
from services.live_stream import LiveStreamAuditor

# Configure structured logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("visionaudit.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for model warm-up and graceful shutdown."""
    logger.info("Starting VisionAudit Microservice...")
    
    # 1. Warm-up YOLOv8 model to eliminate cold-start penalty on first request
    try:
        logger.info(f"Loading and warming up YOLOv8 weights from '{settings.yolo_model_path}'...")
        _ = vision_service.model
        logger.info("YOLOv8 model loaded successfully.")
    except Exception as e:
        logger.warning(f"Could not warm up YOLO model on startup: {e}. Model will load on demand.")

    # 2. Check Gemini API key presence
    if not settings.google_api_key:
        logger.warning(
            "GOOGLE_API_KEY is not set in environment or .env file. "
            "LLM reasoning requests will fail until an API key is provided."
        )
    else:
        logger.info("Google Gemini API credentials detected.")

    yield

    logger.info("Shutting down VisionAudit Microservice...")


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production AI microservice combining Ultralytics YOLOv8 (Computer Vision) "
        "and LangChain LCEL with Gemini 1.5-flash for automated facility safety, "
        "inventory, and operational risk audits."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for microservice cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Fallback handler to prevent unhandled 500 stack trace leaks."""
    logger.error(f"Unhandled system exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "An internal system error occurred.",
            "detail": str(exc) if settings.debug else "Internal Server Error"
        }
    )


# Health check endpoints
@app.get("/health", tags=["Health & Status"], summary="Microservice liveness check")
async def health_check():
    """Health check endpoint for Docker container orchestration and load balancers."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": settings.app_version,
        "yolo_loaded": vision_service._model is not None,
        "gemini_configured": bool(settings.google_api_key)
    }


@app.get("/dashboard", tags=["Dashboard"], summary="Live Browser Visual Telemetry & Audit Dashboard")
async def get_dashboard():
    return FileResponse("live_dashboard.html")


@app.get("/", tags=["Health & Status"], summary="Root service discovery")
async def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "dashboard": "/dashboard",
        "audit_endpoint": f"{settings.api_v1_prefix}/audit/analyze-frame",
        "live_audit_websocket": "/ws/live-audit"
    }


# Mount API routers
app.include_router(audit_router, prefix=settings.api_v1_prefix)


# Root-level live audit WebSocket alias
@app.websocket("/ws/live-audit")
async def live_audit_ws_root(
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
    """Direct root live audit WebSocket endpoint (/ws/live-audit)."""
    await websocket.accept()
    auditor = LiveStreamAuditor(
        websocket=websocket,
        source=source,
        include_preview=preview,
        audit_interval=interval,
        critical_cooldown=cooldown
    )
    await auditor.run()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)
