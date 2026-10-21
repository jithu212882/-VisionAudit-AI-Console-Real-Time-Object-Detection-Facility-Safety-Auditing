"""
LangChain LCEL & Gemini Reasoning Audit Tests.
Verifies prompt construction, JsonOutputParser schema validation, detection details formatting,
and report structure compliance.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.audit import (
    BoundingBox,
    VisionTelemetry,
    FacilityAuditReport,
    AuditFinding,
    FindingCategory,
    SeverityLevel
)
from services.reasoning import reasoning_service


def test_reasoning_service_initialization():
    assert reasoning_service is not None
    assert hasattr(reasoning_service, "_parser")
    assert reasoning_service.temperature == 0.2


def test_format_detection_details():
    detections = [
        BoundingBox(class_id=0, class_name="person", confidence=0.945, box=[50.0, 60.0, 200.0, 300.0]),
        BoundingBox(class_id=63, class_name="laptop", confidence=0.881, box=[250.0, 150.0, 450.0, 350.0])
    ]
    formatted = reasoning_service._format_detection_details(detections)
    assert "person" in formatted
    assert "0.95" in formatted or "0.94" in formatted
    assert "laptop" in formatted
    assert "BBox" in formatted


def test_empty_detection_details():
    formatted = reasoning_service._format_detection_details([])
    assert "No object detections" in formatted


def test_facility_audit_report_schema_validation():
    sample_data = {
        "timestamp": "2026-10-21T12:00:00Z",
        "total_detections": 2,
        "summary": "Monitored area shows standard operations with active staff member and workstation.",
        "findings": [
            {
                "category": "Safety",
                "observation": "Clear passage maintained along workstation perimeter.",
                "severity": "LOW",
                "recommended_action": "Continue regular shift audits."
            },
            {
                "category": "Inventory",
                "observation": "Laptop logged on desk.",
                "severity": "LOW",
                "recommended_action": "Record asset tag."
            }
        ]
    }
    report = FacilityAuditReport.model_validate(sample_data)
    assert report.total_detections == 2
    assert len(report.findings) == 2
    assert report.findings[0].category == FindingCategory.SAFETY
    assert report.findings[1].category == FindingCategory.INVENTORY


if __name__ == "__main__":
    print("[*] Running LangChain Reasoning Audit Tests...")
    test_reasoning_service_initialization()
    test_format_detection_details()
    test_empty_detection_details()
    test_facility_audit_report_schema_validation()
    print("[+] All LangChain Reasoning Audit tests passed successfully!")
