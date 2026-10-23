"""
Automated Integration Tests for Face Recognition Access Control & Door Simulator
Tests:
1. GET /admin/users - List enrolled personnel
2. POST /admin/register-face - Enroll new employee
3. POST /verify-access - Verify authorized personnel (confidence >= 0.6)
4. POST /verify-access - Deny intruder personnel (confidence < 0.6)
5. GET /door-demo - Verify HTML frontend simulator serving
6. DELETE /admin/users/{name} - Clean up enrolled personnel
"""

import numpy as np
import cv2
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def create_synthetic_face_image(seed=42):
    """Generates a synthetic portrait image with distinct facial features for test stability."""
    np.random.seed(seed)
    canvas = np.full((160, 160, 3), 190, dtype=np.uint8)
    
    # Head & shoulders
    cv2.circle(canvas, (80, 65), 32, (60, 70, 80), -1)
    cv2.ellipse(canvas, (80, 150), (48, 40), 0, 180, 360, (40, 50, 60), -1)
    
    # Eyes & nose
    cv2.circle(canvas, (70, 60), 4, (20, 20, 20), -1)
    cv2.circle(canvas, (90, 60), 4, (20, 20, 20), -1)
    cv2.line(canvas, (80, 63), (80, 72), (30, 30, 30), 2)
    
    # Random texture fingerprint
    noise = np.random.randint(-15, 15, canvas.shape, dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    _, buffer = cv2.imencode(".jpg", canvas)
    return buffer.tobytes()


def test_face_access_control_flow():
    print("--- 1. Testing GET /admin/users ---")
    res = client.get("/admin/users")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    initial_users = res.json()
    assert isinstance(initial_users, list)
    print(f"Initial enrolled users count: {len(initial_users)}")

    print("\n--- 2. Testing POST /admin/register-face ---")
    test_name = "Inspector Jane Doe"
    test_role = "Chief Security Architect"
    face_bytes = create_synthetic_face_image(seed=101)

    reg_res = client.post(
        "/admin/register-face",
        data={"name": test_name, "role": test_role},
        files={"file": ("jane_doe.jpg", face_bytes, "image/jpeg")}
    )
    assert reg_res.status_code == 200, f"Expected 200, got {reg_res.status_code}: {reg_res.text}"
    reg_data = reg_res.json()
    assert reg_data["status"] == "success"
    assert reg_data["name"] == test_name
    assert reg_data["embedding_dimension"] == 128
    print(f"Registered user: {reg_data['name']} (Dim: {reg_data['embedding_dimension']})")

    print("\n--- 3. Testing POST /verify-access for Authorized Personnel ---")
    verify_auth_res = client.post(
        "/verify-access",
        files={"file": ("jane_probe.jpg", face_bytes, "image/jpeg")}
    )
    assert verify_auth_res.status_code == 200
    auth_data = verify_auth_res.json()
    print("Authorized Response:", auth_data)
    assert auth_data["access_granted"] is True
    assert auth_data["matched_name"] == test_name
    assert auth_data["confidence"] >= 0.60
    assert "timestamp" in auth_data

    print("\n--- 4. Testing POST /verify-access for Intruder / Unknown ---")
    intruder_canvas = np.zeros((160, 160, 3), dtype=np.uint8)
    intruder_canvas[::8, :] = [255, 0, 50]
    intruder_canvas[:, ::8] = [0, 200, 255]
    _, intruder_buf = cv2.imencode(".jpg", intruder_canvas)
    intruder_bytes = intruder_buf.tobytes()

    verify_intruder_res = client.post(
        "/verify-access",
        files={"file": ("intruder.jpg", intruder_bytes, "image/jpeg")}
    )
    assert verify_intruder_res.status_code == 200
    intruder_data = verify_intruder_res.json()
    print("Intruder Response:", intruder_data)
    assert intruder_data["access_granted"] is False
    assert intruder_data["matched_name"] == "Unknown"
    assert intruder_data["confidence"] < 0.60


    print("\n--- 5. Testing GET /door-demo ---")
    demo_res = client.get("/door-demo")
    assert demo_res.status_code == 200
    assert "3D Door Simulator" in demo_res.text
    print("Door simulator HTML demo served successfully.")

    print("\n--- 6. Testing DELETE /admin/users/{name} ---")
    del_res = client.delete(f"/admin/users/{test_name}")
    assert del_res.status_code == 200
    del_data = del_res.json()
    assert del_data["status"] == "success"
    print(f"Deleted user: {test_name}")

    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")


if __name__ == "__main__":
    test_face_access_control_flow()
