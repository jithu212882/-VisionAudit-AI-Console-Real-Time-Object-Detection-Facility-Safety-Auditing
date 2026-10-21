from enum import Enum
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, AliasChoices, field_validator


class FindingCategory(str, Enum):
    SAFETY = "Safety"
    INVENTORY = "Inventory"
    OPERATIONAL_RISK = "Operational Risk"


class SeverityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class AuditFinding(BaseModel):
    category: FindingCategory = Field(
        default=FindingCategory.OPERATIONAL_RISK,
        description="Category of the audit finding: Safety, Inventory, or Operational Risk"
    )
    observation: str = Field(
        default="Operational area observed during facility audit.",
        description="Detailed observation derived from detected visual telemetry and patterns",
        validation_alias=AliasChoices("observation", "finding", "description")
    )
    severity: SeverityLevel = Field(
        default=SeverityLevel.LOW,
        description="Assessed severity level: LOW, MEDIUM, or HIGH"
    )
    recommended_action: str = Field(
        default="Inspect zone for safety and protocol compliance.",
        description="Specific actionable corrective or preventative recommendation",
        validation_alias=AliasChoices("recommended_action", "recommendation", "action")
    )

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, v):
        if not v or not isinstance(v, str):
            return SeverityLevel.LOW
        v_upper = v.strip().upper()
        if v_upper in ("LOW", "MEDIUM", "HIGH"):
            return SeverityLevel(v_upper)
        return SeverityLevel.LOW

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v):
        if not v or not isinstance(v, str):
            return FindingCategory.OPERATIONAL_RISK
        v_clean = v.strip().title()
        if "Safety" in v_clean:
            return FindingCategory.SAFETY
        elif "Inventory" in v_clean:
            return FindingCategory.INVENTORY
        return FindingCategory.OPERATIONAL_RISK


class FacilityAuditReport(BaseModel):
    timestamp: str = Field(
        ...,
        description="ISO 8601 or formatted timestamp of the audit analysis",
        validation_alias=AliasChoices("timestamp", "time", "date")
    )
    total_detections: int = Field(
        default=0,
        description="Total count of all objects detected by the vision model",
        validation_alias=AliasChoices("total_detections", "total_count", "total_entities")
    )
    summary: str = Field(
        default="Facility operations audit generated based on visual detection telemetry.",
        description="High-level executive summary of facility observations and operational posture",
        validation_alias=AliasChoices("summary", "executive_summary", "overview")
    )
    findings: List[AuditFinding] = Field(
        default=[],
        description="List of specific policy audit findings",
        validation_alias=AliasChoices("findings", "audit_findings", "issues")
    )


class BoundingBox(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    box: List[float] = Field(
        ..., 
        description="Normalized or pixel bounding box coordinates [x1, y1, x2, y2]"
    )


class VisionTelemetry(BaseModel):
    total_count: int
    class_counts: Dict[str, int]
    detections: List[BoundingBox]
    annotated_image_base64: Optional[str] = Field(
        default=None, 
        description="Base64 encoded JPEG representation of the annotated frame"
    )


class VisionAuditResponse(BaseModel):
    status: str = "success"
    telemetry: VisionTelemetry
    report: FacilityAuditReport


class StreamEventType(str, Enum):
    TELEMETRY = "telemetry"
    CRITICAL_ALERT = "critical_alert"
    PERIODIC_AUDIT = "periodic_audit"
    STATUS = "status"
    ERROR = "error"


class LiveStreamMessage(BaseModel):
    event: StreamEventType = Field(
        ...,
        description="Type of live stream event (telemetry, critical_alert, periodic_audit, status, error)"
    )
    timestamp: str = Field(..., description="ISO 8601 timestamp of message dispatch")
    fps: Optional[float] = Field(default=None, description="Current camera stream inference FPS")
    telemetry: Optional[VisionTelemetry] = Field(
        default=None,
        description="Real-time detection telemetry and class occurrences"
    )
    preview: Optional[str] = Field(
        default=None,
        description="Base64 encoded JPEG annotated preview frame"
    )
    report: Optional[FacilityAuditReport] = Field(
        default=None,
        description="Throttled LLM FacilityAuditReport (included on alerts or periodic audits)"
    )
    trigger_reason: Optional[str] = Field(
        default=None,
        description="Explanation of why the LLM audit was triggered (e.g. Critical Object Detected: person)"
    )
    message: Optional[str] = Field(
        default=None,
        description="Human-readable informational or error message"
    )

