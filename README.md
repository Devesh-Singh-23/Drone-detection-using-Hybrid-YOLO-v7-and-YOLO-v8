# Tactical Drone Surveillance System

Real-time drone detection and classification using a **YOLOv7 → YOLOv8** cascaded deep learning pipeline with a military-themed PyQt5 GUI featuring tactical radar, threat assessment, and audio alerts.

<!-- Add your screenshot here -->
![Surveillance GUI](assets/Screenshot%202026-05-05%20015220.png)

---

## Features

- **Two-Stage Pipeline** — YOLOv7 detects drones in the frame, YOLOv8 classifies each detection into drone type
- **Tactical GUI** — Military-themed PyQt5 interface with live video feed, bounding box overlays, and confidence readouts
- **Radar Sweep** — Animated tactical radar that maps detected drone positions as threat-colored blips
- **Threat Classification** — Automatic threat level assignment based on drone model:
  | Drone Type | Threat Level | Color |
  |---|---|---|
  | DJI Inspire | 🔴 HIGH | Red |
  | DJI Mavic | 🟠 MEDIUM | Orange |
  | DJI Phantom | 🟡 LOW | Yellow |
  | No Drone | ⚫ NONE | Grey |
- **Audio Alerts** — Dual-tone beep on first detection with cooldown logic to prevent alarm fatigue
- **Real-Time Performance** — 20+ FPS on NVIDIA RTX 3050 via FP16 mixed-precision inference

---

## Demo

<!-- Add your demo images/GIFs here -->
![Detection Example](assets/Screenshot%202026-05-18%20013857.png)

---

## Tech Stack

| Component | Technology |
|---|---|
| Detection | YOLOv7x (custom-trained) |
| Classification | YOLOv8m-cls (fine-tuned on synthetic drone dataset) |
| GUI | PyQt5 with custom-painted radar widget |
| GPU Acceleration | PyTorch CUDA + FP16 + cuDNN benchmark |
| Video Capture | OpenCV |

---

## Project Structure

```
Drone3/
├── drone_surveillance.py      # Main application — full tactical GUI
├── train_yolov8_cls.py        # YOLOv8 classifier training script
├── requirements.txt           # Python dependencies
├── LICENSE                    # MIT License
├── README.md                  # This file
├── assets/                    # Screenshots and demo images
├── weights/                   # Trained model weights (included via Git LFS)
│   ├── yolov7_best.pt         # YOLOv7 detector weights
│   └── yolov8_cls_best.pt     # YOLOv8 classifier weights
└── yolov7/                    # YOLOv7 repository (not tracked — see setup)
```

---

## Setup

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (tested on RTX 3050)
- Webcam
- [Git LFS](https://git-lfs.github.com/) (for downloading model weights)

### 1. Clone the Repository

```bash
git lfs install
git clone https://github.com/YOUR_USERNAME/Drone3.git
cd Drone3
```

> **Note:** `git lfs install` ensures the model weights (~165 MB) are downloaded properly.

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA (adjust cu118 to match your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

### 4. Set Up YOLOv7

```bash
# Clone YOLOv7 into the project directory
git clone https://github.com/WongKinYiu/yolov7.git
```

Model weights are **included in the repo** (via Git LFS) in the `weights/` directory — no manual download needed.

---

## Usage

### Run the Surveillance System

```bash
python drone_surveillance.py
```

This launches the full tactical GUI with:
- Live webcam feed with bounding boxes and threat labels
- Tactical radar with sweep animation and detection blips
- System status panel (target count, threat level, FPS, confidence scores)
- Audio alert on drone detection

**Press the window close button (✕) to exit.**

### Train the Classifier

```bash
python train_yolov8_cls.py
```

Trains a YOLOv8 classification model on the synthetic drone dataset with aggressive augmentation (AugMix, random erasing, mixup) to bridge the synthetic-to-real domain gap.

---

## How It Works

```
Webcam Frame
     │
     ▼
┌─────────────────────┐
│  YOLOv7 Detection   │  → Bounding boxes around drones
│  (640×640, FP16)     │
└─────────┬───────────┘
          │ Crop each detection
          ▼
┌─────────────────────┐
│  YOLOv8 Classifier  │  → Drone type + confidence
│  (224×224)           │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Tactical GUI       │  → Video overlay + Radar + Alerts
│  (PyQt5)            │
└─────────────────────┘
```

1. **Detection** — YOLOv7 processes each frame at 640×640 resolution, producing bounding boxes with confidence scores
2. **Classification** — Each detected region is cropped, resized to 224×224, and classified by YOLOv8 into one of 4 classes
3. **Threat Mapping** — Classification results are mapped to threat levels (HIGH/MEDIUM/LOW/NONE)
4. **Visualization** — Results are rendered on the GUI with threat-colored bounding boxes, radar blips, and status indicators

---

## Output

<!-- Add your output screenshots here -->

### Tactical GUI with Active Detection
![GUI Active Detection](assets/surveillance_demo.png)

### Radar with Multiple Targets
![Radar View](assets/radar_view.png)

---

## Configuration

Key parameters can be adjusted in `drone_surveillance.py`:

```python
DETECT_IMG_SIZE   = 640     # YOLOv7 inference resolution
CLASSIFY_IMG_SIZE = 224     # YOLOv8 input size
CONF_THRESHOLD    = 0.25    # Detection confidence threshold
IOU_THRESHOLD     = 0.45    # NMS IoU threshold
WEBCAM_INDEX      = 0       # Camera device index
ALERT_COOLDOWN_SEC = 3.0    # Seconds before alert can re-trigger
```

---

## Dataset

The classifier is trained on the **Synthetic Drone Classification Dataset** with the following structure:

```
Synthetic_Drone_Classification_Dataset/
├── train/
│   ├── dji_inspire/
│   ├── dji_mavic/
│   ├── dji_phantom/
│   └── no_drone/
└── val/
    ├── dji_inspire/
    ├── dji_mavic/
    ├── dji_phantom/
    └── no_drone/
```

Training uses aggressive augmentation to handle the synthetic-to-real domain gap:
- HSV jitter (hue ±3%, saturation ±90%, brightness ±60%)
- Rotation (±15°), translation (±20%), scale (±70%)
- Random erasing (50%), mixup (10%), AugMix

---

## Hardware Tested

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 (4 GB VRAM) |
| OS | Windows 10/11 |
| Camera | Built-in laptop webcam (640×480) |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
