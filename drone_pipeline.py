"""
Real-Time Drone Detection & Classification Pipeline
=====================================================
Stage 1 → YOLOv7 object detector  : finds drones, produces bounding boxes
Stage 2 → YOLOv8 classifier       : classifies each crop into drone type

Requirements (already in .venv):
    torch, torchvision, opencv-python, numpy, ultralytics
"""

import sys
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

# ---------------------------------------------------------------------------
# YOLOv7 lives in a local repo; add it to sys.path so we can import its utils
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
YOLOV7_DIR = os.path.join(PROJECT_ROOT, "yolov7")
sys.path.insert(0, YOLOV7_DIR)

from models.experimental import attempt_load                       # YOLOv7 model loader
from utils.general import non_max_suppression, scale_coords, check_img_size  # NMS + coord helpers
from utils.datasets import letterbox                               # resize + pad
from utils.torch_utils import select_device, time_synchronized     # device + timing

from ultralytics import YOLO  # YOLOv8 classifier (ultralytics API)

# ============================= CONFIGURATION ================================
YOLOV7_WEIGHTS = os.path.join(YOLOV7_DIR, "runs", "train", "yolov7x_cpu", "weights", "best.pt")
YOLOV8_CLS_WEIGHTS = os.path.join(PROJECT_ROOT, "runs", "classify", "drone_classification_v3_medium", "weights", "best.pt")

DETECT_IMG_SIZE = 640       # YOLOv7 inference resolution (pixels)
CLASSIFY_IMG_SIZE = 224     # YOLOv8 classifier input size (must match training)

CONF_THRESHOLD = 0.25       # Detection confidence threshold
IOU_THRESHOLD = 0.45        # NMS IoU threshold

WEBCAM_INDEX = 0            # Laptop camera
# ============================================================================


# ───────────────────────── 1. Model Loading ─────────────────────────────────

def load_models():
    """
    Load YOLOv7 detection model and YOLOv8 classification model.
    Automatically selects GPU if available, otherwise CPU.
    Returns:
        detector      – YOLOv7 model (PyTorch)
        classifier    – YOLOv8 model (ultralytics)
        device        – torch.device used
        stride        – model stride (needed for letterbox padding)
        half          – whether FP16 is active
        class_names   – dict of YOLOv8 classifier class names
    """
    # Device selection — use RTX 3050
    device = select_device("0")  # NVIDIA GeForce RTX 3050
    half = device.type != "cpu"  # FP16 only on CUDA
    print(f"[INFO] Using device: {device}")

    # ── YOLOv7 Detector ──
    try:
        detector = attempt_load(YOLOV7_WEIGHTS, map_location=device)
        stride = int(detector.stride.max())
        if half:
            detector.half()
        # Warmup – run a dummy tensor through the model once
        warmup_tensor = torch.zeros(1, 3, DETECT_IMG_SIZE, DETECT_IMG_SIZE).to(device)
        if half:
            warmup_tensor = warmup_tensor.half()
        detector(warmup_tensor)
        print(f"[INFO] YOLOv7 detector loaded  ({YOLOV7_WEIGHTS})")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLOv7 model: {e}")
        sys.exit(1)

    # ── YOLOv8 Classifier ──
    try:
        classifier = YOLO(YOLOV8_CLS_WEIGHTS)
        # Run one dummy inference to warm up + discover class names
        dummy = np.zeros((CLASSIFY_IMG_SIZE, CLASSIFY_IMG_SIZE, 3), dtype=np.uint8)
        dummy_results = classifier(dummy, verbose=False, device=0)
        class_names = dummy_results[0].names  # e.g. {0: 'dji_inspire', 1: 'dji_mavic', …}
        print(f"[INFO] YOLOv8 classifier loaded ({YOLOV8_CLS_WEIGHTS})")
        print(f"[INFO] Classification classes : {class_names}")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLOv8 model: {e}")
        sys.exit(1)

    return detector, classifier, device, stride, half, class_names


# ───────────────────────── 2. Frame Preprocessing ───────────────────────────

def preprocess_frame(frame, img_size, stride, device, half):
    """
    Prepare a raw BGR webcam frame for YOLOv7 inference.
    Steps:
        1. Letterbox resize to `img_size` while keeping aspect ratio.
        2. BGR → RGB, HWC → CHW.
        3. Normalize 0-255 → 0.0-1.0.
        4. Add batch dimension.
    Returns:
        tensor  – preprocessed image tensor (1, 3, H, W)
        frame   – original (unmodified) frame for drawing later
    """
    img = letterbox(frame, img_size, stride=stride, auto=True)[0]  # resized + padded
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR → RGB, HWC → CHW
    img = np.ascontiguousarray(img)

    tensor = torch.from_numpy(img).to(device)
    tensor = tensor.half() if half else tensor.float()
    tensor /= 255.0  # normalize
    if tensor.ndimension() == 3:
        tensor = tensor.unsqueeze(0)  # add batch dim

    return tensor


# ───────────────────────── 3. Drone Detection (YOLOv7) ──────────────────────

def detect_drones(detector, tensor, original_shape, device, conf_thres, iou_thres):
    """
    Run YOLOv7 inference + NMS on a preprocessed tensor.

    Args:
        detector        – YOLOv7 model
        tensor          – preprocessed image tensor (1, 3, H, W)
        original_shape  – shape of the original frame (H, W, C)
        device          – torch.device
        conf_thres      – confidence threshold for NMS
        iou_thres       – IoU threshold for NMS

    Returns:
        detections – list of [x1, y1, x2, y2, confidence] in original-frame coords.
                     Empty list if nothing detected.
    """
    with torch.no_grad():
        pred = detector(tensor)[0]

    # Non-Max Suppression → list[ Tensor(n, 6) ]  where 6 = (x1,y1,x2,y2,conf,cls)
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    detections = []
    for det in pred:
        if len(det):
            # Rescale boxes from inference-tensor coords → original frame coords
            det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], original_shape).round()

            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = [int(v.item()) for v in xyxy]
                detections.append((x1, y1, x2, y2, float(conf.item())))

    return detections


# ───────────────────────── 4. Drone Classification (YOLOv8) ─────────────────

def classify_drone(classifier, frame, bbox, class_names):
    """
    Crop the detected drone region and classify it.

    Args:
        classifier   – YOLOv8 classification model
        frame        – original BGR frame
        bbox         – (x1, y1, x2, y2, det_conf)
        class_names  – dict {idx: name}

    Returns:
        class_name   – predicted drone type string
        cls_conf     – classification confidence (0-1)
    """
    x1, y1, x2, y2, _ = bbox
    h, w = frame.shape[:2]

    # Clamp coordinates to frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Guard against degenerate crops
    if (x2 - x1) < 5 or (y2 - y1) < 5:
        return "unknown", 0.0

    crop = frame[y1:y2, x1:x2]

    # Resize to classifier input size (224×224)
    crop_resized = cv2.resize(crop, (CLASSIFY_IMG_SIZE, CLASSIFY_IMG_SIZE),
                              interpolation=cv2.INTER_LINEAR)

    # Run YOLOv8 classification inference (handles normalization internally)
    with torch.no_grad():
        results = classifier(crop_resized, verbose=False, device=1)

    probs = results[0].probs
    top1_idx = int(probs.top1)
    cls_conf = float(probs.top1conf.item())
    class_name = class_names[top1_idx]

    return class_name, cls_conf


# ───────────────────────── 5. Result Drawing ────────────────────────────────

def draw_results(frame, detections, classifications, fps):
    """
    Overlay bounding boxes, class labels, confidence scores, and FPS on frame.

    Args:
        frame           – original BGR frame (modified in-place)
        detections      – list of (x1, y1, x2, y2, det_conf)
        classifications – list of (class_name, cls_conf) aligned with detections
        fps             – current FPS value
    """
    drone_count = 0

    for bbox, (cls_name, cls_conf) in zip(detections, classifications):
        x1, y1, x2, y2, det_conf = bbox

        # Choose color based on class
        if cls_name == "no_drone":
            color = (0, 0, 255)      # Red for "no_drone" (false positive)
        else:
            color = (0, 255, 0)      # Green for actual drone
            drone_count += 1

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label: "class_name  det:0.85  cls:0.92"
        label = f"{cls_name}  det:{det_conf:.2f}  cls:{cls_conf:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y1, label_size[1] + 10)

        # Label background
        cv2.rectangle(frame,
                      (x1, label_y - label_size[1] - 6),
                      (x1 + label_size[0] + 4, label_y + baseline - 2),
                      color, -1)
        cv2.putText(frame, label, (x1 + 2, label_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # ── HUD: drone count + FPS ──
    cv2.putText(frame, f"Drones: {drone_count}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

    if drone_count == 0 and len(detections) == 0:
        cv2.putText(frame, "No drones detected", (15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 1, cv2.LINE_AA)


# ───────────────────────── 6. Main Loop ─────────────────────────────────────

def main():
    print("=" * 60)
    print("  Drone Detection + Classification Pipeline")
    print("  YOLOv7 (detect)  →  YOLOv8 (classify)")
    print("=" * 60)

    # Load models
    detector, classifier, device, stride, half, class_names = load_models()

    # Validate inference resolution against model stride
    img_size = check_img_size(DETECT_IMG_SIZE, s=stride)

    # Enable cuDNN benchmark for constant-size inputs
    cudnn.benchmark = True

    # Open webcam
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check camera index / permissions.")
        sys.exit(1)

    # Optionally set resolution (uncomment for lower-latency capture)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"[INFO] Webcam opened (index {WEBCAM_INDEX}). Press 'q' to quit.\n")

    fps = 0.0
    frame_count = 0

    try:
        while True:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame. Retrying...")
                continue

            # ── Stage 1: Detection ──
            tensor = preprocess_frame(frame, img_size, stride, device, half)
            detections = detect_drones(detector, tensor, frame.shape,
                                       device, CONF_THRESHOLD, IOU_THRESHOLD)

            # ── Stage 2: Classification (run on each detected region) ──
            classifications = []
            for bbox in detections:
                cls_name, cls_conf = classify_drone(classifier, frame, bbox, class_names)
                classifications.append((cls_name, cls_conf))

            # ── Draw overlays ──
            draw_results(frame, detections, classifications, fps)

            # Show the annotated frame
            cv2.imshow("Drone Pipeline  |  Press 'q' to exit", frame)

            # FPS calculation (exponential moving average for smooth display)
            elapsed = time.time() - t_start
            current_fps = 1.0 / max(elapsed, 1e-6)
            fps = fps * 0.9 + current_fps * 0.1 if frame_count > 0 else current_fps
            frame_count += 1

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released. Pipeline stopped.")


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
