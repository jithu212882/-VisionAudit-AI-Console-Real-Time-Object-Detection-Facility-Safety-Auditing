"""
Quick Verification Script for ReasoningService
Validates LangChain LCEL + Gemini output parsing with Pydantic v2 schemas.
"""

import asyncio
from schemas.audit import VisionTelemetry, BoundingBox
from services.reasoning import reasoning_service


async def main():
    print("[*] Running Quick Reasoning Audit Test...")
    
    mock_telemetry = VisionTelemetry(
        total_count=3,
        class_counts={"person": 1, "chair": 2},
        detections=[
            BoundingBox(class_id=0, class_name="person", confidence=0.88, box=[100.0, 150.0, 250.0, 400.0]),
            BoundingBox(class_id=56, class_name="chair", confidence=0.74, box=[300.0, 200.0, 380.0, 350.0]),
            BoundingBox(class_id=56, class_name="chair", confidence=0.69, box=[400.0, 210.0, 490.0, 360.0]),
        ]
    )

    try:
        report = await reasoning_service.audit_telemetry(mock_telemetry, "2026-10-21T11:00:00Z")
        print("\n[+] SUCCESS! FacilityAuditReport Generated:")
        print(f"Summary: {report.summary}")
        print(f"Total Detections: {report.total_detections}")
        print(f"Findings ({len(report.findings)}):")
        for f in report.findings:
            print(f"  - [{f.category}] ({f.severity}) {f.observation}")
            print(f"    Action: {f.recommended_action}")
    except Exception as e:
        print(f"\n[-] ERROR: {e}")


if __name__ == "__main__":
    asyncio.run(main())
