from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Explicitly resolve absolute path to local .env file
ENV_PATH = Path(__file__).resolve().parent / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
else:
    load_dotenv(find_dotenv(), override=True)


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""
    
    app_name: str = "VisionAudit Microservice"
    app_version: str = "1.0.0"
    debug: bool = False
    
    api_v1_prefix: str = "/api/v1"
    
    # Computer Vision Settings
    yolo_model_path: str = "yolov8n.pt"
    yolo_confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    
    # LangChain / Google Gemini Settings
    google_api_key: str = Field(default="", description="Google Gemini API Key")
    gemini_model: str = "gemini-1.5-flash"
    gemini_temperature: float = 0.2
    
    # Live Streaming & LLM Throttling Settings
    live_audit_interval_seconds: float = Field(
        default=15.0, 
        description="Interval in seconds between periodic background audits"
    )
    critical_alert_cooldown_seconds: float = Field(
        default=10.0, 
        description="Minimum seconds to wait between critical event LLM triggers"
    )
    critical_object_classes: list[str] = Field(
        default=["person", "fire", "knife", "scissors"],
        description="Object classes that immediately trigger safety/security audit checks"
    )
    
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH) if ENV_PATH.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

# Immediate verification confirmation
print(f"[VisionAudit] Gemini API Key Loaded: {'YES' if settings.google_api_key else 'NO'}")
