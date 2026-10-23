# VisionAudit AI Microservice

A production-grade AI microservice combining **Ultralytics YOLOv8** (Computer Vision) and **LangChain LCEL** with **Gemini 1.5-flash** (Structured Policy & Compliance Auditing) inside a Dockerized **FastAPI** backend.

---

## 🏗 System Architecture

```mermaid
graph LR
    Client([Client / Camera Feed]) -->|Multipart POST /api/v1/audit/analyze-frame| FastAPIRouter[FastAPI Audit Router]
    
    subgraph Vision Layer
        FastAPIRouter -->|Raw Image Bytes| ThreadPool[Thread Pool / asyncio.to_thread]
        ThreadPool -->|Decode & Predict| YOLO[Ultralytics YOLOv8n (conf >= 0.4)]
        YOLO -->|Extract BBoxes & Frequencies| Telemetry[Vision Telemetry]
        YOLO -->|results.plot| Annotate[Base64 Encoded JPEG]
    end

    subgraph Reasoning Layer LCEL
        Telemetry -->|Telemetry Dict & Timestamp| Prompt[ChatPromptTemplate]
        Prompt -->|Pipe Operator '|' | Gemini[ChatGoogleGenerativeAI gemini-1.5-flash]
        Gemini -->|with_structured_output| Schema[FacilityAuditReport Pydantic v2]
    end

    Schema --> UnifiedResp[VisionAuditResponse JSON]
    Annotate --> UnifiedResp
    UnifiedResp -->|200 OK| Client
```

---

## 📁 Repository Directory Structure

```text
d:/OPEN CV PROJECT/
├── config.py                 # Pydantic-settings configuration (.env loader)
├── main.py                   # FastAPI application initialization & lifespan
├── requirements.txt          # Production pinned dependencies
├── Dockerfile                # Multi-stage optimized Debian slim image
├── .dockerignore             # Docker build context exclusions
├── .gitignore                # Git repository exclusions (.env, virtualenvs)
├── .env.example              # Sample environment configuration template
├── test_client.py            # Verification and test client for HTTP POST
├── test_live_stream.py       # Live WebSocket client with real-time CLI monitor
├── test_desktop_view.py      # Standalone desktop OpenCV live feed with green bounding boxes
├── live_dashboard.html       # Web browser dashboard for real-time live video audit (/dashboard)
├── schemas/
│   ├── __init__.py           # Schema package export
│   └── audit.py              # Pydantic v2 data models & WebSocket message schemas
├── services/
│   ├── __init__.py           # Services package export
│   ├── vision.py             # YOLOv8 inference, telemetry extraction, and annotation
│   ├── reasoning.py          # LangChain modern LCEL chain with Gemini 1.5-flash
│   └── live_stream.py        # Continuous camera streaming & throttled audit engine
└── routers/
    ├── __init__.py           # Router package export
    └── audit.py              # APIRouter (/api/v1/audit/analyze-frame & /ws/live-audit)
```

---

## 🚀 Quick Start Guide

### 1. Environment Configuration

Copy `.env.example` to `.env` and configure your Google Gemini API key:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
GOOGLE_API_KEY=AIzaSy...your_gemini_api_key
YOLO_MODEL_PATH=yolov8n.pt
YOLO_CONFIDENCE_THRESHOLD=0.4
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TEMPERATURE=0.2
DEBUG=false
```

### 2. Local Execution (Virtual Environment)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server with Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🐳 Docker Deployment

### Build Container

```bash
docker build -t visionaudit:1.0.0 .
```

### Run Container

```bash
docker run -d \
  --name visionaudit \
  -p 8000:8000 \
  -e GOOGLE_API_KEY="your_gemini_api_key_here" \
  visionaudit:1.0.0
```

Container logs:
```bash
docker logs -f visionaudit
```

---

## 📡 API Reference

### Health Check

`GET /health`

```json
{
  "status": "healthy",
  "service": "VisionAudit Microservice",
  "version": "1.0.0",
  "yolo_loaded": true,
  "gemini_configured": true
}
```

---

### Frame Audit Endpoint

`POST /api/v1/audit/analyze-frame`

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: Image file binary (JPEG, PNG, WebP) *(Required)*
  - `timestamp`: ISO-8601 string *(Optional)*

#### Example cURL:

```bash
curl -X POST "http://localhost:8000/api/v1/audit/analyze-frame" \
  -H "Accept: application/json" \
  -F "file=@warehouse_floor.jpg;type=image/jpeg" \
  -F "timestamp=2026-10-21T09:44:00Z"
```

#### Example Response Body:

```json
{
  "status": "success",
  "telemetry": {
    "total_count": 5,
    "class_counts": {
      "person": 2,
      "chair": 2,
      "laptop": 1
    },
    "detections": [
      {
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.8842,
        "box": [112.5, 95.2, 230.1, 440.8]
      }
    ],
    "annotated_image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
  },
  "report": {
    "timestamp": "2026-10-21T09:44:00Z",
    "total_detections": 5,
    "summary": "Audit of the designated zone reveals 2 active personnel in proximity to workstation assets with compliant layout and minimal operational risk.",
    "findings": [
      {
        "category": "Safety",
        "observation": "Personnel detected within standard operating distance of computer terminals with unobstructed clear path.",
        "severity": "LOW",
        "recommended_action": "Maintain scheduled routine inspections of aisleway clearance."
      },
      {
        "category": "Inventory",
        "observation": "Identified 2 chairs and 1 laptop asset logged within workstation sector.",
        "severity": "LOW",
        "recommended_action": "Log active asset allocation in facility inventory database."
      }
    ]
  }
}
```

---

### Live WebSocket Stream Endpoint

`WS /ws/live-audit?source=0&preview=false&interval=10.0&cooldown=5.0`

Connect via WebSocket to run continuous YOLOv8 inference and receive live telemetry with throttled LangChain Gemini 1.5-flash audit reports.

- **Query Parameters**:
  - `source`: Webcam index (`0`) or RTSP URL (`rtsp://user:pass@camera-ip:554/stream`). Default: `"0"`.
  - `preview`: `true` to include Base64 annotated frame in telemetry (default: `false`).
  - `interval`: Interval in seconds between periodic summary audits (default: `10.0`).
  - `cooldown`: Minimum seconds between critical event alerts (default: `5.0`).

#### Stream Message Types:
1. `status`: Stream connection status.
2. `telemetry`: High-frequency real-time YOLOv8 detections (FPS, counts, bounding boxes).
3. `critical_alert`: Immediate alert when critical entities (e.g. `person`, `fire`) appear.
4. `periodic_audit`: Periodic facility compliance summary report.
5. `error`: Stream error or camera disconnect notification.

#### Testing Live Stream:

**CLI Client:**
```bash
python test_live_stream.py 0
```

**Browser Dashboard:**
Open `http://localhost:8000/dashboard` (or `live_dashboard.html`) in your browser to monitor the live feed with green bounding boxes, detection tags, and a real-time HUD overlay.

**Native Desktop OpenCV Window:**
```bash
python test_desktop_view.py --source 0
```
*Press 'q' or ESC in the window to exit.*

---

## 🚪 Face Recognition Access Control & 3D Door Simulator

A standalone biometric security access module with admin personnel onboarding, real-time webcam scanning with face bounding boxes, and an interactive CSS 3D animated security door chamber.

### Endpoints
- `GET /door-demo`: Interactive 3-panel browser demo (Admin Enrollment, Live Scanner, 3D Door Simulator & Real-time Audit Logs).
- `POST /admin/register-face`: Enrolls personnel by extracting 128-D ArcFace/SFace facial embeddings and caching them in memory & JSON storage.
- `GET /admin/users`: Returns list of enrolled personnel with avatars, roles, and timestamps.
- `DELETE /admin/users/{name}`: De-registers personnel from the access control database.
- `POST /verify-access`: Compares an incoming cropped face/frame or embedding vector against enrolled personnel using cosine similarity (threshold = 0.60).
  * Returns: `{"access_granted": true/false, "matched_name": "...", "confidence": 0.94, "timestamp": "..."}`.

### Run Automated Tests
```bash
python test_access_control.py
```


