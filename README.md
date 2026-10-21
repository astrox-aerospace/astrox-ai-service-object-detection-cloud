# ASTROX Vision Microservice (Autonomous Aerial Perception & Inference Engine)

The **ASTROX Vision Microservice** is a high-throughput, low-latency computer vision subsystem designed for real-time aerial reconnaissance and target tracking on unmanned aerial vehicles (UAVs). Built upon **FastAPI**, **Ultralytics YOLO**, and **ByteTrack**, the service handles multi-protocol video ingestion, hardware-accelerated deep learning inference, and low-latency streaming to human operators.

---

## 1. System Role & Architecture Overview

Within the ASTROX platform, this service operates as **Backend 2 (The Vision Engine)**. It is decoupled from the web application layer to ensure dedicated compute allocation for neural network workloads and to facilitate independent deployment on GPU cloud nodes or onboard edge computers (e.g., NVIDIA Jetson).

### System Integration Flow

```mermaid
flowchart LR
    subgraph Capture["Video Ingestion"]
        RTSP["RTSP Stream (Drone Datalink)"]
        CSI["CSI Camera (nvarguscamerasrc)"]
        USB["USB V4L2 Sensor"]
    end

    subgraph Core["Vision Microservice (This Service :8000)"]
        INGEST["GStreamer / OpenCV Capture"]
        INFER["YOLO Detection Engine\n(TensorRT / CUDA)"]
        TRACK["ByteTrack Association\n(Kalman Filter)"]
        INGEST --> INFER --> TRACK
    end

    subgraph Egress["Egress & Clients"]
        GATEWAY["Express Application Gateway\n(:5000)"]
        CONSOLE["Operator Console\n(:3000)"]
    end

    Capture --> INGEST
    GATEWAY -->|Synchronous Image Scan| INFER
    INFER -->|Detection Payloads| GATEWAY
    TRACK -.->|WebRTC Video Feed| CONSOLE
    TRACK -.->|WebSocket Telemetry (Bounding Boxes)| CONSOLE
```

---

## 2. Related System Repositories

| Repository | Primary Stack | Architecture Role | Description |
| :--- | :--- | :--- | :--- |
| [**Drone-Object-Detection-Website**](https://github.com/Mahavidyajp/Drone-Object-Detection-Website) | React, Node.js, Express, MongoDB | Application Layer | Operator command dashboard, user management, telemetry persistence, and API proxy. |
| [**astrox-cloud-deploy**](https://github.com/Mahavidyajp/astrox-cloud-deploy) | Docker, Docker Compose, Nginx | Orchestration Hub | Central cloud deployment definitions, environment templates, and reverse proxy routing. |

---

## 3. Core Capabilities

* **Hardware-Accelerated Inference:** Native execution of YOLOv8 and YOLO11 architectures utilizing TensorRT FP16/INT8 engines on NVIDIA Jetson and CUDA GPUs.
* **Persistent Multi-Target Tracking:** Integrates ByteTrack with motion prediction via Kalman filters, maintaining identity tracking through partial occlusions and high-speed maneuvers.
* **Low-Latency Streaming:** Distributes real-time annotated video through WebRTC peer connections and low-latency WebSocket bounding box metadata channels.
* **Multi-Sensor Ingestion:** GStreamer pipelines optimized for CSI cameras (`nvarguscamerasrc`), RTSP network streams, and USB V4L2 sensors.
* **Hardware Telemetry:** Monitors thermal thresholds, GPU utilization, and memory consumption via `/system/stats`.

---

## 4. Setup & Deployment Runbooks

### Option A: Standard Linux Host (CUDA GPU or CPU)

#### 1. Clone the repository
```bash
git clone https://github.com/Mahavidyajp/astrox-ai-service.git
cd astrox-ai-service
```

#### 2. Initialize Python virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Launch the vision microservice
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

---

### Option B: NVIDIA Jetson Embedded Platform (JetPack 6 / L4T r36.x)

To deploy on an onboard NVIDIA Jetson (Orin Nano / Xavier NX) without shadowing vendor-provided JetPack packages:

```bash
# 1. Ensure system dependencies and build headers are configured
./setup-jetson.sh

# 2. Inherit system site-packages to utilize JetPack-optimized PyTorch/TensorRT
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

# 3. Install top-level application packages without replacing JetPack libraries
pip install --no-deps "fastapi>=0.110" "uvicorn[standard]>=0.29" "pydantic>=2.6" "PyYAML>=6.0" "ultralytics>=8.2"

# 4. Start service with GPU acceleration
python3 app.py
```

---

## 5. API Reference Specification

### System & Health Endpoints
| Endpoint | Method | Response | Description |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | `{"status": "healthy"}` | Probes service status and model initialization. |
| `/model/classes` | `GET` | Array of strings | Returns all class labels supported by the active model. |
| `/system/stats` | `GET` | JSON payload | Telemetry for CPU, GPU utilization, memory, and core temperature. |

### Batch & Media Inference Endpoints
| Endpoint | Method | Payload | Description |
| :--- | :--- | :--- | :--- |
| `/detect` | `POST` | Multipart Form (`file`) | Analyzes single image frame; returns detected classes, confidence scores, and bounding box coordinates `[x1, y1, x2, y2]`. |
| `/video/detect` | `POST` | Multipart Form (`file`) | Executes frame-by-frame inference on uploaded video payloads; yields annotated video download artifacts. |

### Live Surveillance & Streaming Endpoints
| Endpoint | Method | Protocol | Description |
| :--- | :--- | :--- | :--- |
| `/webcam/cameras` | `GET` | REST | Scans and enumerates available local video capture devices. |
| `/webcam/rtsp/connect` | `POST` | REST | Establishes connection to an external RTSP video feed (e.g., airborne drone datalink). |
| `/webcam/webrtc/offer` | `POST` | REST / SDP | Negotiates SDP offer/answer for real-time WebRTC video streaming. |
| `/webcam/ws/detections` | `GET` | WebSocket | High-frequency stream delivering target tracking bounding boxes and telemetry. |

---

## 6. Configuration Parameters

* **`bytetrack.yaml`**: Configures tracker hyperparameters including `track_thresh` (initialization threshold), `match_thresh` (association gate), and `track_buffer` (frame retention during occlusions).
* **`cameras.yaml.example`**: Specification template for static multi-camera arrays and RTSP addresses.

---

## 7. Verification & Smoke Testing

Execute the following commands to confirm operational integrity:

```bash
# 1. Health check
curl -s -f http://localhost:8000/health

# 2. Single-frame inference smoke test
curl -X POST "http://localhost:8000/detect" \
     -H "Accept: application/json" \
     -F "file=@sample_frame.jpg"
```

---
