"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           TACTICAL DRONE SURVEILLANCE SYSTEM  ·  v3.0                      ║
║           YOLOv7 Detection  →  YOLOv8 Classification                       ║
║           Army Hackathon Demo  ·  Defense-Grade Pipeline                    ║
║           + Tactical Radar Sweep Animation                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Requirements:
    PyQt5, torch, torchvision, opencv-python, numpy, ultralytics, winsound
"""

import sys
import os
import time
import math
import threading

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPointF, QRectF
from PyQt5.QtGui import (QImage, QPixmap, QFont, QColor, QPalette, QPainter,
                         QPen, QBrush, QRadialGradient, QConicalGradient)

# ── Platform-specific sound ──
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# ---------------------------------------------------------------------------
# YOLOv7 local repo imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
YOLOV7_DIR = os.path.join(PROJECT_ROOT, "yolov7")
sys.path.insert(0, YOLOV7_DIR)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.datasets import letterbox
from utils.torch_utils import select_device

from ultralytics import YOLO

# ============================= CONFIGURATION ================================
YOLOV7_WEIGHTS   = os.path.join(PROJECT_ROOT, "weights", "yolov7_best.pt")
YOLOV8_CLS_WEIGHTS = os.path.join(PROJECT_ROOT, "weights", "yolov8_cls_best.pt")

DETECT_IMG_SIZE   = 640
CLASSIFY_IMG_SIZE = 224
CONF_THRESHOLD    = 0.25
IOU_THRESHOLD     = 0.45
WEBCAM_INDEX      = 0

ALERT_COOLDOWN_SEC = 3.0
BLIP_FADE_SEC      = 3.0   # seconds before radar blips fade out
RADAR_SWEEP_MS     = 30    # sweep timer interval (ms)
SWEEP_SPEED        = 2.0   # degrees per tick
# ============================================================================

# ── Threat level mapping ──
THREAT_MAP = {
    "dji_inspire":  {"level": "HIGH",   "color": (0, 0, 255),    "hex": "#FF0000",
                     "radar_color": QColor(255, 30, 30)},
    "dji_mavic":    {"level": "MEDIUM", "color": (0, 165, 255),  "hex": "#FF8C00",
                     "radar_color": QColor(255, 140, 0)},
    "dji_phantom":  {"level": "LOW", "color": (0, 200, 255),  "hex": "#FFC800",
                     "radar_color": QColor(255, 220, 0)},
    "no_drone":     {"level": "NONE",   "color": (80, 80, 80),   "hex": "#505050",
                     "radar_color": QColor(80, 80, 80)},
    "unknown":      {"level": "LOW",    "color": (128, 128, 128), "hex": "#808080",
                     "radar_color": QColor(80, 200, 80)},
}


# ═══════════════════════════════════════════════════════════════════════════════
#  1. MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_models():
    """
    Load YOLOv7 detector + YOLOv8 classifier on GPU (RTX 3050).
    Returns (detector, classifier, device, stride, half, class_names).
    """
    device = select_device("0")
    half = device.type != "cpu"
    print(f"[SYSTEM] Device: {device}")

    # ── YOLOv7 ──
    try:
        detector = attempt_load(YOLOV7_WEIGHTS, map_location=device)
        stride = int(detector.stride.max())
        if half:
            detector.half()
        warmup = torch.zeros(1, 3, DETECT_IMG_SIZE, DETECT_IMG_SIZE).to(device)
        if half:
            warmup = warmup.half()
        detector(warmup)
        print("[SYSTEM] YOLOv7 detector  → READY")
    except Exception as e:
        print(f"[FATAL] YOLOv7 load failed: {e}")
        sys.exit(1)

    # ── YOLOv8 ──
    try:
        classifier = YOLO(YOLOV8_CLS_WEIGHTS)
        dummy = np.zeros((CLASSIFY_IMG_SIZE, CLASSIFY_IMG_SIZE, 3), dtype=np.uint8)
        dummy_res = classifier(dummy, verbose=False, device=0)
        class_names = dummy_res[0].names
        print(f"[SYSTEM] YOLOv8 classifier → READY  classes={class_names}")
    except Exception as e:
        print(f"[FATAL] YOLOv8 load failed: {e}")
        sys.exit(1)

    return detector, classifier, device, stride, half, class_names


# ═══════════════════════════════════════════════════════════════════════════════
#  2. DETECTION (YOLOv7)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_frame(frame, img_size, stride, device, half):
    """Letterbox + normalize → tensor for YOLOv7."""
    img = letterbox(frame, img_size, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    tensor = torch.from_numpy(img).to(device)
    tensor = tensor.half() if half else tensor.float()
    tensor /= 255.0
    if tensor.ndimension() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def detect_drones(detector, tensor, orig_shape, conf_thres, iou_thres):
    """Run YOLOv7 inference + NMS. Returns list of (x1,y1,x2,y2,conf)."""
    with torch.no_grad():
        pred = detector(tensor)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    detections = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], orig_shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = [int(v.item()) for v in xyxy]
                detections.append((x1, y1, x2, y2, float(conf.item())))
    return detections


# ═══════════════════════════════════════════════════════════════════════════════
#  3. CLASSIFICATION (YOLOv8)
# ═══════════════════════════════════════════════════════════════════════════════

def classify_drone(classifier, frame, bbox, class_names):
    """Crop detection region → classify → return (class_name, confidence)."""
    x1, y1, x2, y2, _ = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if (x2 - x1) < 5 or (y2 - y1) < 5:
        return "unknown", 0.0

    crop = frame[y1:y2, x1:x2]
    crop_resized = cv2.resize(crop, (CLASSIFY_IMG_SIZE, CLASSIFY_IMG_SIZE),
                              interpolation=cv2.INTER_LINEAR)

    with torch.no_grad():
        results = classifier(crop_resized, verbose=False, device=0)

    probs = results[0].probs
    top1_idx = int(probs.top1)
    cls_conf = float(probs.top1conf.item())
    return class_names[top1_idx], cls_conf


# ═══════════════════════════════════════════════════════════════════════════════
#  4. ALERT SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class AlertSystem:
    """
    Manages audible + visual alerts with cooldown logic.
    - Triggers ONCE per detection event.
    - Resets only after no drones for ALERT_COOLDOWN_SEC seconds.
    """

    def __init__(self):
        self.alert_active = False
        self.last_drone_time = 0.0
        self.blink_state = False

    def update(self, drone_detected: bool):
        now = time.time()
        if drone_detected:
            self.last_drone_time = now
            if not self.alert_active:
                self.alert_active = True
                self._play_sound()
            return True
        else:
            if self.alert_active and (now - self.last_drone_time) > ALERT_COOLDOWN_SEC:
                self.alert_active = False
            return self.alert_active

    def _play_sound(self):
        def _beep():
            if HAS_WINSOUND:
                winsound.Beep(1200, 300)
                winsound.Beep(1500, 200)
            else:
                print("\a")
        threading.Thread(target=_beep, daemon=True).start()

    def toggle_blink(self):
        self.blink_state = not self.blink_state


# ═══════════════════════════════════════════════════════════════════════════════
#  5. RADAR WIDGET
# ═══════════════════════════════════════════════════════════════════════════════

class RadarWidget(QWidget):
    """
    Tactical radar sweep animation panel.

    Features:
        - Circular radar grid with range rings and bearing lines
        - Smooth rotating sweep line with conical gradient trail
        - Threat-colored blips derived from bounding box center positions
        - Blips fade out over BLIP_FADE_SEC seconds
        - 30ms timer drives the sweep (independent of YOLO inference)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Sweep angle (degrees, 0 = north, clockwise)
        self.sweep_angle = 0.0

        # Blip storage: list of {x_norm, y_norm, threat, timestamp}
        # x_norm, y_norm ∈ [0, 1] — normalized position in frame
        self.blips = []

        # Frame dimensions (updated by detection pipeline)
        self.frame_w = 640
        self.frame_h = 480

        # Sweep timer — runs independently of inference
        self.sweep_timer = QTimer(self)
        self.sweep_timer.timeout.connect(self._update_sweep)
        self.sweep_timer.start(RADAR_SWEEP_MS)

    # ─────────────── Public API ───────────────

    def add_blip(self, x_center, y_center, threat_level, frame_w, frame_h):
        """
        Add a radar blip from a detection's bounding box center.

        Args:
            x_center, y_center – pixel center of bbox in original frame
            threat_level       – string: "HIGH", "MEDIUM", "LOW", "NONE"
            frame_w, frame_h   – frame dimensions for normalization
        """
        self.frame_w = frame_w
        self.frame_h = frame_h

        # Normalize to [0, 1]
        x_norm = x_center / max(frame_w, 1)
        y_norm = y_center / max(frame_h, 1)

        self.blips.append({
            "x": x_norm,
            "y": y_norm,
            "threat": threat_level,
            "time": time.time(),
        })

    def clear_and_update(self, detections, classifications, frame_shape):
        """
        Bulk-update blips from current frame's detections.
        Old blips are kept until they fade; new ones are added.
        """
        h, w = frame_shape[:2]
        for bbox, (cls_name, cls_conf) in zip(detections, classifications):
            if cls_name == "no_drone":
                continue
            x1, y1, x2, y2, _ = bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            threat = THREAT_MAP.get(cls_name, THREAT_MAP["unknown"])["level"]
            self.add_blip(cx, cy, threat, w, h)

    def fade_blips(self):
        """Remove blips older than BLIP_FADE_SEC."""
        now = time.time()
        self.blips = [b for b in self.blips if (now - b["time"]) < BLIP_FADE_SEC]

    # ─────────────── Sweep Timer ───────────────

    def _update_sweep(self):
        """Advance sweep angle and trigger repaint."""
        self.sweep_angle = (self.sweep_angle + SWEEP_SPEED) % 360.0
        self.fade_blips()
        self.update()  # triggers paintEvent

    # ─────────────── Painting ───────────────

    def paintEvent(self, event):
        """Custom paint: radar grid + sweep + blips."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w = self.width()
        h = self.height()
        size = min(w, h) - 4
        radius = size / 2.0
        cx = w / 2.0
        cy = h / 2.0

        # ── Background ──
        painter.fillRect(self.rect(), QColor(8, 8, 8))

        # ── Outer circle border ──
        painter.setPen(QPen(QColor(0, 255, 65, 100), 2))
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # ── Range rings (3 concentric circles at 25%, 50%, 75%) ──
        painter.setPen(QPen(QColor(0, 255, 65, 40), 1))
        for frac in (0.25, 0.50, 0.75):
            r = radius * frac
            painter.drawEllipse(QPointF(cx, cy), r, r)

        # ── Bearing lines (every 45°) ──
        painter.setPen(QPen(QColor(0, 255, 65, 35), 1))
        for angle_deg in range(0, 360, 45):
            rad = math.radians(angle_deg)
            x2 = cx + radius * math.sin(rad)
            y2 = cy - radius * math.cos(rad)
            painter.drawLine(QPointF(cx, cy), QPointF(x2, y2))

        # ── Cardinal labels ──
        painter.setPen(QPen(QColor(0, 255, 65, 120), 1))
        font = QFont("Consolas", 8)
        painter.setFont(font)
        label_r = radius + 2
        for label, angle_deg in [("N", 0), ("E", 90), ("S", 180), ("W", 270)]:
            rad = math.radians(angle_deg)
            lx = cx + label_r * math.sin(rad) - 4
            ly = cy - label_r * math.cos(rad) + 4
            painter.drawText(QPointF(lx, ly), label)

        # ── Sweep trail (conical gradient for glow effect) ──
        sweep_rad = math.radians(self.sweep_angle)
        trail_span = 40  # degrees of glow trail behind sweep

        # Draw filled sector for the sweep trail
        painter.setPen(Qt.NoPen)
        gradient = QConicalGradient(cx, cy, -(self.sweep_angle - 90))
        gradient.setColorAt(0.0, QColor(0, 255, 65, 60))
        gradient.setColorAt(trail_span / 360.0, QColor(0, 255, 65, 0))
        gradient.setColorAt(1.0, QColor(0, 255, 65, 0))
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # ── Sweep line ──
        sx = cx + radius * math.sin(sweep_rad)
        sy = cy - radius * math.cos(sweep_rad)
        painter.setPen(QPen(QColor(0, 255, 65, 180), 2))
        painter.drawLine(QPointF(cx, cy), QPointF(sx, sy))

        # ── Blips ──
        now = time.time()
        for blip in self.blips:
            age = now - blip["time"]
            if age >= BLIP_FADE_SEC:
                continue

            # Fade: 1.0 → 0.0 over BLIP_FADE_SEC
            alpha = max(0.0, 1.0 - (age / BLIP_FADE_SEC))

            # Map normalized frame coords to radar circle
            # Frame center → radar center, frame edges → radar edge
            # x: 0→1 maps to -1→+1, y: 0→1 maps to -1→+1
            bx_rel = (blip["x"] - 0.5) * 2.0   # [-1, +1]
            by_rel = (blip["y"] - 0.5) * 2.0   # [-1, +1]

            # Clamp to circle
            dist = math.sqrt(bx_rel ** 2 + by_rel ** 2)
            if dist > 1.0:
                bx_rel /= dist
                by_rel /= dist

            bx = cx + bx_rel * radius * 0.9
            by = cy + by_rel * radius * 0.9

            # Color based on threat level
            threat = blip["threat"]
            if threat == "HIGH":
                base_color = QColor(255, 30, 30)
            elif threat == "MEDIUM":
                base_color = QColor(255, 200, 0)
            else:
                base_color = QColor(0, 255, 65)

            # ── Blip glow (radial gradient) ──
            glow_r = 12
            glow = QRadialGradient(QPointF(bx, by), glow_r)
            glow_color = QColor(base_color)
            glow_color.setAlpha(int(100 * alpha))
            glow.setColorAt(0.0, glow_color)
            glow_color_edge = QColor(base_color)
            glow_color_edge.setAlpha(0)
            glow.setColorAt(1.0, glow_color_edge)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(QPointF(bx, by), glow_r, glow_r)

            # ── Blip dot ──
            dot_color = QColor(base_color)
            dot_color.setAlpha(int(255 * alpha))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(dot_color))
            blip_size = 4 + 2 * alpha  # shrink as fading
            painter.drawEllipse(QPointF(bx, by), blip_size, blip_size)

            # ── Blip ring (pulse effect for fresh blips) ──
            if age < 0.5:
                ring_alpha = int(150 * (1.0 - age / 0.5))
                ring_color = QColor(base_color)
                ring_color.setAlpha(ring_alpha)
                painter.setPen(QPen(ring_color, 1.5))
                painter.setBrush(Qt.NoBrush)
                ring_r = 6 + 10 * (age / 0.5)
                painter.drawEllipse(QPointF(bx, by), ring_r, ring_r)

        # ── Center dot ──
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 255, 65, 200)))
        painter.drawEllipse(QPointF(cx, cy), 3, 3)

        painter.end()


# ═══════════════════════════════════════════════════════════════════════════════
#  6. VIDEO WORKER THREAD
# ═══════════════════════════════════════════════════════════════════════════════

class VideoWorker(QThread):
    """
    Runs the detection + classification pipeline in a background thread
    and emits processed frames + metadata to the GUI.
    """
    frame_ready = pyqtSignal(np.ndarray, list, list, float)

    def __init__(self, detector, classifier, device, stride, half, class_names):
        super().__init__()
        self.detector = detector
        self.classifier = classifier
        self.device = device
        self.stride = stride
        self.half = half
        self.class_names = class_names
        self.img_size = check_img_size(DETECT_IMG_SIZE, s=stride)
        self.running = True
        self.fps = 0.0

    def run(self):
        cudnn.benchmark = True
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            print("[FATAL] Cannot open webcam.")
            return

        frame_count = 0
        while self.running:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                continue

            # ── Stage 1: Detect ──
            tensor = preprocess_frame(frame, self.img_size, self.stride,
                                      self.device, self.half)
            detections = detect_drones(self.detector, tensor, frame.shape,
                                       CONF_THRESHOLD, IOU_THRESHOLD)

            # ── Stage 2: Classify each detection ──
            classifications = []
            for bbox in detections:
                cls_name, cls_conf = classify_drone(self.classifier, frame,
                                                     bbox, self.class_names)
                classifications.append((cls_name, cls_conf))

            # FPS (exponential moving average)
            elapsed = time.time() - t0
            current_fps = 1.0 / max(elapsed, 1e-6)
            self.fps = self.fps * 0.9 + current_fps * 0.1 if frame_count > 0 else current_fps
            frame_count += 1

            self.frame_ready.emit(frame, detections, classifications, self.fps)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# ═══════════════════════════════════════════════════════════════════════════════
#  7. MAIN GUI WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class SurveillanceWindow(QMainWindow):
    """
    Military-themed drone surveillance GUI with tactical radar.
    Layout:
        ┌────────────────────────────┬───────────────┐
        │                            │  ┌─────────┐  │
        │                            │  │  RADAR  │  │
        │     LIVE VIDEO FEED        │  │  SWEEP  │  │
        │     (bounding boxes +      │  └─────────┘  │
        │      threat overlays)      │  STATUS PANEL │
        │                            │  (count, fps, │
        │                            │   threat,     │
        │                            │   alerts)     │
        └────────────────────────────┴───────────────┘
    """

    BG_DARK      = "#0A0A0A"
    BG_PANEL     = "#111111"
    BORDER_GREEN = "#00FF41"
    TEXT_GREEN   = "#00FF41"
    TEXT_DIM     = "#4A6A4A"
    TEXT_WHITE   = "#D0D0D0"
    ALERT_RED    = "#FF1A1A"
    HEADER_BG    = "#0D1B0D"

    def __init__(self, detector, classifier, device, stride, half, class_names):
        super().__init__()
        self.class_names = class_names
        self.alert_system = AlertSystem()

        self._init_ui()

        # ── Video worker ──
        self.worker = VideoWorker(detector, classifier, device, stride, half, class_names)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.start()

        # ── Blink timer (500 ms toggle for alert text) ──
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.alert_system.toggle_blink)
        self.blink_timer.start(500)

    # ─────────────── UI Construction ───────────────

    def _init_ui(self):
        self.setWindowTitle("⬡ TACTICAL DRONE SURVEILLANCE SYSTEM  ·  v3.0")
        self.setMinimumSize(1400, 780)
        self.setStyleSheet(f"background-color: {self.BG_DARK};")

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ── Left: Video feed ──
        video_container = QFrame()
        video_container.setStyleSheet(f"""
            QFrame {{
                border: 2px solid {self.BORDER_GREEN};
                border-radius: 4px;
                background-color: #000000;
            }}
        """)
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(2, 2, 2, 2)

        self.video_label = QLabel("INITIALIZING FEED...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(f"color: {self.TEXT_GREEN}; font-size: 18px;"
                                       f" border: none;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout.addWidget(self.video_label)

        main_layout.addWidget(video_container, stretch=7)

        # ── Right: Panel with Radar + Info ──
        panel = QFrame()
        panel.setFixedWidth(340)
        panel.setStyleSheet(f"""
            QFrame {{
                border: 2px solid {self.BORDER_GREEN};
                border-radius: 4px;
                background-color: {self.BG_PANEL};
            }}
        """)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(6)

        # ── Radar Widget (top of panel) ──
        radar_header = QLabel("◆ TACTICAL RADAR ◆")
        radar_header.setAlignment(Qt.AlignCenter)
        radar_header.setStyleSheet(f"""
            color: {self.TEXT_GREEN};
            font-size: 12px;
            font-weight: bold;
            font-family: Consolas, monospace;
            padding: 4px;
            background-color: {self.HEADER_BG};
            border: 1px solid {self.BORDER_GREEN};
            border-radius: 3px;
        """)
        panel_layout.addWidget(radar_header)

        self.radar = RadarWidget()
        self.radar.setFixedHeight(280)
        self.radar.setStyleSheet("border: none;")
        panel_layout.addWidget(self.radar)

        panel_layout.addWidget(self._make_separator())

        # ── System Status Header ──
        status_header = QLabel("◆ SYSTEM STATUS ◆")
        status_header.setAlignment(Qt.AlignCenter)
        status_header.setStyleSheet(f"""
            color: {self.TEXT_GREEN};
            font-size: 12px;
            font-weight: bold;
            font-family: Consolas, monospace;
            padding: 4px;
            background-color: {self.HEADER_BG};
            border: 1px solid {self.BORDER_GREEN};
            border-radius: 3px;
        """)
        panel_layout.addWidget(status_header)

        # Status indicator
        self.status_label = self._make_info_label("STATUS", "SCANNING")
        panel_layout.addWidget(self.status_label)

        # Drone count
        self.count_label = self._make_info_label("TARGETS", "0")
        panel_layout.addWidget(self.count_label)

        # Highest threat
        self.threat_label = self._make_info_label("THREAT LEVEL", "NONE")
        panel_layout.addWidget(self.threat_label)

        # Drone type
        self.type_label = self._make_info_label("DRONE TYPE", "—")
        panel_layout.addWidget(self.type_label)

        panel_layout.addWidget(self._make_separator())

        # Confidence row
        conf_row = QHBoxLayout()
        self.det_conf_label = self._make_info_label("DET", "—")
        self.cls_conf_label = self._make_info_label("CLS", "—")
        conf_row.addWidget(self.det_conf_label)
        conf_row.addWidget(self.cls_conf_label)
        conf_widget = QWidget()
        conf_widget.setLayout(conf_row)
        conf_widget.setStyleSheet("border: none;")
        panel_layout.addWidget(conf_widget)

        # FPS + GPU row
        perf_row = QHBoxLayout()
        self.fps_label = self._make_info_label("FPS", "0.0")
        self.gpu_label = self._make_info_label("GPU", "RTX 3050")
        perf_row.addWidget(self.fps_label)
        perf_row.addWidget(self.gpu_label)
        perf_widget = QWidget()
        perf_widget.setLayout(perf_row)
        perf_widget.setStyleSheet("border: none;")
        panel_layout.addWidget(perf_widget)

        panel_layout.addWidget(self._make_separator())

        # Alert banner
        self.alert_banner = QLabel("")
        self.alert_banner.setAlignment(Qt.AlignCenter)
        self.alert_banner.setStyleSheet(f"""
            color: {self.ALERT_RED};
            font-size: 18px;
            font-weight: bold;
            font-family: Consolas, monospace;
            padding: 8px;
            border: 2px solid transparent;
            border-radius: 4px;
        """)
        self.alert_banner.setWordWrap(True)
        panel_layout.addWidget(self.alert_banner)

        panel_layout.addStretch()

        # Footer
        footer = QLabel("TACTICAL DRONE SURVEILLANCE v3.0")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet(f"""
            color: {self.TEXT_DIM};
            font-size: 9px;
            font-family: Consolas, monospace;
            border: none;
        """)
        panel_layout.addWidget(footer)

        main_layout.addWidget(panel, stretch=0)

    def _make_info_label(self, title, value):
        lbl = QLabel(f"<span style='color:{self.TEXT_DIM}; font-size:10px;'>"
                     f"{title}</span><br>"
                     f"<span style='color:{self.TEXT_GREEN}; font-size:16px; "
                     f"font-weight:bold;'>{value}</span>")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(f"""
            font-family: Consolas, monospace;
            padding: 4px;
            border: 1px solid #1A3A1A;
            border-radius: 3px;
            background-color: #0A140A;
        """)
        return lbl

    def _update_info_label(self, label, title, value, color=None):
        c = color or self.TEXT_GREEN
        label.setText(f"<span style='color:{self.TEXT_DIM}; font-size:10px;'>"
                      f"{title}</span><br>"
                      f"<span style='color:{c}; font-size:16px; "
                      f"font-weight:bold;'>{value}</span>")

    def _make_separator(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #1A3A1A; max-height: 1px; border: none;")
        return sep

    # ─────────────── Frame Processing ───────────────

    def _on_frame(self, frame, detections, classifications, fps):
        """Slot: receives processed frame + data from worker thread."""

        real_drones = [(d, c) for d, c in zip(detections, classifications)
                       if c[0] != "no_drone"]
        drone_count = len(real_drones)
        drone_detected = drone_count > 0

        # ── Update radar blips ──
        self.radar.clear_and_update(detections, classifications, frame.shape)

        # ── Alert logic ──
        show_alert = self.alert_system.update(drone_detected)

        # ── Draw bounding boxes + labels on frame ──
        for bbox, (cls_name, cls_conf) in zip(detections, classifications):
            x1, y1, x2, y2, det_conf = bbox
            threat = THREAT_MAP.get(cls_name, THREAT_MAP["unknown"])
            color = threat["color"]

            thickness = 3 if threat["level"] in ("HIGH", "MEDIUM") else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Corner accents
            corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)

            # Label
            label = f"{cls_name.upper()} [{threat['level']}]"
            conf_text = f"D:{det_conf:.0%} C:{cls_conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - th - 18), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, conf_text, (x1 + 4, y1 - th - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # ── Tactical crosshair ──
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        cross_color = (0, 255, 65)
        cv2.line(frame, (cx - 20, cy), (cx - 8, cy), cross_color, 1)
        cv2.line(frame, (cx + 8, cy), (cx + 20, cy), cross_color, 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy - 8), cross_color, 1)
        cv2.line(frame, (cx, cy + 8), (cx, cy + 20), cross_color, 1)
        cv2.circle(frame, (cx, cy), 4, cross_color, 1)

        # ── Timestamp + LIVE ──
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "LIVE", (w - 60, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # ── Convert to QPixmap ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio,
                               Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

        # ── Update side panel ──
        self._update_panel(drone_count, real_drones, fps, show_alert)

    def _update_panel(self, drone_count, real_drones, fps, show_alert):
        # Status
        if drone_count > 0:
            self._update_info_label(self.status_label, "STATUS", "⚠ ACTIVE",
                                    self.ALERT_RED)
        else:
            self._update_info_label(self.status_label, "STATUS", "SCANNING",
                                    self.TEXT_GREEN)

        # Target count
        count_color = self.ALERT_RED if drone_count > 0 else self.TEXT_GREEN
        self._update_info_label(self.count_label, "TARGETS", str(drone_count),
                                count_color)

        # Highest threat + drone type
        if real_drones:
            priority = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}
            best = max(real_drones,
                       key=lambda x: priority.get(
                           THREAT_MAP.get(x[1][0], THREAT_MAP["unknown"])["level"], 0))
            best_bbox, (best_cls, best_cls_conf) = best
            threat_info = THREAT_MAP.get(best_cls, THREAT_MAP["unknown"])

            self._update_info_label(self.threat_label, "THREAT LEVEL",
                                    threat_info["level"], threat_info["hex"])
            self._update_info_label(self.type_label, "DRONE TYPE",
                                    best_cls.upper().replace("_", " "),
                                    threat_info["hex"])
            self._update_info_label(self.det_conf_label, "DET",
                                    f"{best_bbox[4]:.0%}", self.TEXT_GREEN)
            self._update_info_label(self.cls_conf_label, "CLS",
                                    f"{best_cls_conf:.0%}", self.TEXT_GREEN)
        else:
            self._update_info_label(self.threat_label, "THREAT LEVEL", "NONE",
                                    self.TEXT_DIM)
            self._update_info_label(self.type_label, "DRONE TYPE", "—",
                                    self.TEXT_DIM)
            self._update_info_label(self.det_conf_label, "DET", "—", self.TEXT_DIM)
            self._update_info_label(self.cls_conf_label, "CLS", "—", self.TEXT_DIM)

        # FPS
        fps_color = (self.TEXT_GREEN if fps >= 20
                     else "#FFA500" if fps >= 10
                     else self.ALERT_RED)
        self._update_info_label(self.fps_label, "FPS", f"{fps:.1f}", fps_color)

        # Alert banner (blinking)
        if show_alert:
            if self.alert_system.blink_state:
                self.alert_banner.setText("⚠ DRONE DETECTED ⚠")
                self.alert_banner.setStyleSheet(f"""
                    color: {self.ALERT_RED};
                    font-size: 18px;
                    font-weight: bold;
                    font-family: Consolas, monospace;
                    padding: 8px;
                    border: 2px solid {self.ALERT_RED};
                    border-radius: 4px;
                    background-color: #1A0000;
                """)
            else:
                self.alert_banner.setText("⚠ DRONE DETECTED ⚠")
                self.alert_banner.setStyleSheet(f"""
                    color: #660000;
                    font-size: 18px;
                    font-weight: bold;
                    font-family: Consolas, monospace;
                    padding: 8px;
                    border: 2px solid #330000;
                    border-radius: 4px;
                    background-color: {self.BG_PANEL};
                """)
        else:
            self.alert_banner.setText("ALL CLEAR")
            self.alert_banner.setStyleSheet(f"""
                color: {self.TEXT_DIM};
                font-size: 14px;
                font-family: Consolas, monospace;
                padding: 8px;
                border: 1px solid #1A3A1A;
                border-radius: 4px;
                background-color: {self.BG_PANEL};
            """)

    # ─────────────── Cleanup ───────────────

    def closeEvent(self, event):
        self.worker.stop()
        self.blink_timer.stop()
        self.radar.sweep_timer.stop()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
#  8. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  ◆ TACTICAL DRONE SURVEILLANCE SYSTEM  v3.0 ◆")
    print("  YOLOv7 Detection  →  YOLOv8 Classification")
    print("  Defense-Grade Real-Time Pipeline + Radar")
    print("=" * 65)

    detector, classifier, device, stride, half, class_names = load_models()
    print("[SYSTEM] All models loaded. Launching GUI...\n")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(10, 10, 10))
    palette.setColor(QPalette.WindowText, QColor(0, 255, 65))
    app.setPalette(palette)

    window = SurveillanceWindow(detector, classifier, device, stride, half, class_names)
    window.showMaximized()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
