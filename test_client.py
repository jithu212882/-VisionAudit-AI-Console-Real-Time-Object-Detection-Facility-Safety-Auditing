"""
Verification and Test Client for VisionAudit Microservice
Creates a sample synthetic image (or takes an existing image path), uploads it to
/api/v1/audit/analyze-frame, prints the telemetry and Pydantic report, and saves
the annotated image.
"""

import sys
import base64
import requests
import numpy as np
import cv2

API_URL = "http://localhost:8000/api/v1/audit/analyze-frame"


def generate_synthetic_test_frame() -> bytes:
    """Generates a synthetic frame with shapes and text for basic testing if no image is provided."""
    # Create a 640x480 RGB image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 240
    
    # Draw floor/desk representation
    cv2.rectangle(img, (50, 200), (590, 450), (180, 180, 180), -1)
    
    # Add textual label
    cv2.putText(img, "VisionAudit Mock Facility Zone", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    
    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise RuntimeError("Failed to encode synthetic test frame.")
    return buffer.tobytes()


def test_audit_endpoint(image_path: str = None):
    print(f"[*] Connecting to {API_URL}...")
    
    if image_path:
        print(f"[*] Reading image from: {image_path}")
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    else:
        print("[*] No image provided. Generating synthetic frame...")
        image_bytes = generate_synthetic_test_frame()

    files = {
        "file": ("test_frame.jpg", image_bytes, "image/jpeg")
    }
    data = {
        "timestamp": "2026-10-21T09:44:00Z"
    }

    try:
        response = requests.post(API_URL, files=files, data=data, timeout=60)
        print(f"[*] HTTP Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== [VISION TELEMETRY] ===")
            print(f"Total Detections: {result['telemetry']['total_count']}")
            print(f"Class Counts: {result['telemetry']['class_counts']}")
            
            print("\n=== [FACILITY AUDIT REPORT (LCEL + GEMINI 1.5-FLASH)] ===")
            report = result["report"]
            print(f"Summary: {report['summary']}")
            print(f"\nFindings ({len(report['findings'])}):")
            for idx, finding in enumerate(report["findings"], 1):
                print(f"\n  Finding #{idx}:")
                print(f"    - Category: {finding['category']}")
                print(f"    - Severity: {finding['severity']}")
                print(f"    - Observation: {finding['observation']}")
                print(f"    - Recommendation: {finding['recommended_action']}")

            # Save annotated image if present
            annotated_b64 = result["telemetry"].get("annotated_image_base64")
            if annotated_b64:
                with open("annotated_output.jpg", "wb") as f_out:
                    f_out.write(base64.b64decode(annotated_b64))
                print("\n[+] Saved annotated frame to 'annotated_output.jpg'.")
        else:
            print(f"[-] Error: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        print("[-] Connection failed. Is the FastAPI server running on http://localhost:8000?")


if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    test_audit_endpoint(img_arg)
