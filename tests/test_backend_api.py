"""
Backend Route & Schema Verification Tests for VisionAudit Microservice.
Tests FastAPI endpoints, OpenAPI specs, WebSocket streaming initialization, and Pydantic schemas.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from main import app
from schemas.audit import (
    FindingCategory,
    SeverityLevel,
    AuditFinding,
    FacilityAuditReport,
    BoundingBox,
    VisionTelemetry,
    LiveStreamMessage,
    StreamEventType,
)


def test_health_check_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "VisionAudit" in data["service"]
    assert "version" in data


def test_root_discovery_endpoint():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "docs" in data
    assert "dashboard" in data
    assert "audit_endpoint" in data
    assert "live_audit_websocket" in data


def test_dashboard_route_serves_html():
    client = TestClient(app)
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "VisionAudit" in response.text


def test_pydantic_schema_validations():
    # Test AuditFinding validation and normalization
    finding = AuditFinding(
        category="Safety Policy Violation",
        observation="Worker in restricted perimeter without PPE.",
        severity="high",
        recommended_action="Notify shift manager and clear area."
    )
    assert finding.category == FindingCategory.SAFETY
    assert finding.severity == SeverityLevel.HIGH

    # Test BoundingBox & VisionTelemetry
    box = BoundingBox(
        class_id=0,
        class_name="person",
        confidence=0.92,
        box=[10.0, 20.0, 100.0, 200.0]
    )
    telem = VisionTelemetry(
        total_count=1,
        class_counts={"person": 1},
        detections=[box],
        annotated_image_base64="dGVzdA=="
    )
    assert telem.total_count == 1
    assert telem.class_counts["person"] == 1

    # Test LiveStreamMessage
    msg = LiveStreamMessage(
        event=StreamEventType.TELEMETRY,
        timestamp="2026-10-21T12:00:00Z",
        fps=30.0,
        telemetry=telem,
        preview=telem.annotated_image_base64
    )
    assert msg.event == StreamEventType.TELEMETRY
    assert msg.fps == 30.0
    assert msg.preview == "dGVzdA=="


def test_websocket_stream_initialization():
    client = TestClient(app)
    with client.websocket_connect("/ws/live-audit?source=annotated_output.jpg&preview=true") as ws:
        status_msg = ws.receive_json()
        assert status_msg["event"] == "status"
        assert "connected" in status_msg["message"].lower()

        telemetry_msg = ws.receive_json()
        assert telemetry_msg["event"] == "telemetry"
        assert "telemetry" in telemetry_msg
        assert "preview" in telemetry_msg


if __name__ == "__main__":
    print("[*] Running Backend API & Schema Tests...")
    test_health_check_endpoint()
    test_root_discovery_endpoint()
    test_dashboard_route_serves_html()
    test_pydantic_schema_validations()
    test_websocket_stream_initialization()
    print("[+] All Backend API & Schema tests passed successfully!")
