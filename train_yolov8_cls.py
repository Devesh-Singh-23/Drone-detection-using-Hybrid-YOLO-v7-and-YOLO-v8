import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from ultralytics import YOLO

# ============================= CONFIGURATION ================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to your drone classification dataset (ImageNet-style directory structure)
DATASET_PATH = os.path.join(PROJECT_ROOT, "archive", "Synthetic_Drone_Classification_Dataset")

# Pre-trained base model (download automatically if not present)
BASE_MODEL = "yolov8s-cls.pt"

# Output directory for training runs
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "runs", "classify")
# ============================================================================


def main():
    print("Initializing YOLOv8 classification model...")
    # Load a pretrained YOLOv8 SMALL model (more capacity for generalization)
    model = YOLO(BASE_MODEL)

    print(f"Starting training on dataset: {DATASET_PATH}")
    # Train with stronger augmentation to bridge synthetic-to-real domain gap
    results = model.train(
        data=DATASET_PATH,
        epochs=30,           # More epochs for better generalization
        imgsz=224,
        batch=8,
        workers=4,
        device=0,
        # Aggressive augmentation to handle domain gap
        hsv_h=0.03,          # Stronger hue variation
        hsv_s=0.9,           # Stronger saturation variation
        hsv_v=0.6,           # Stronger brightness variation
        degrees=15,          # Rotation
        translate=0.2,       # More translation
        scale=0.7,           # More scale variation
        flipud=0.1,          # Vertical flip sometimes
        fliplr=0.5,          # Horizontal flip
        erasing=0.5,         # Random erasing (simulates occlusion)
        mixup=0.1,           # Mix images together
        auto_augment="augmix",  # Stronger auto-augment policy
        project=OUTPUT_DIR,
        name="drone_classification_v2",
        exist_ok=True
    )
    print("Training complete!")

if __name__ == "__main__":
    main()
