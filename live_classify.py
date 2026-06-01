import os
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# ============================= CONFIGURATION ================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to trained YOLOv8 classifier weights
YOLOV8_CLS_WEIGHTS = os.path.join(
    PROJECT_ROOT, "runs", "classify", "drone_classification_v2", "weights", "best.pt"
)

WEBCAM_INDEX = 0          # Camera device index
SMOOTHING_FRAMES = 15     # Rolling average window for prediction smoothing
# ============================================================================


def main():
    # Load the trained classification model
    model = YOLO(YOLOV8_CLS_WEIGHTS)

    # Rolling average over N frames to smooth predictions
    prob_history = deque(maxlen=SMOOTHING_FRAMES)

    # Open the camera
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print(f"Camera opened. Smoothing over {SMOOTHING_FRAMES} frames. Press 'q' to quit.")

    # Get class names from first dummy inference
    dummy_ret, dummy_frame = cap.read()
    dummy_results = model(dummy_frame, verbose=False)
    class_names = dummy_results[0].names  # {0: 'dji_inspire', 1: 'dji_mavic', ...}
    num_classes = len(class_names)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Run classification inference
        results = model(frame, verbose=False)
        result = results[0]
        probs = result.probs

        # Add current frame's probabilities to history
        current_probs = probs.data.cpu().numpy()
        prob_history.append(current_probs)

        # Compute averaged probabilities across recent frames
        avg_probs = np.mean(prob_history, axis=0)
        top1_idx = int(np.argmax(avg_probs))
        top1_conf = avg_probs[top1_idx]
        class_name = class_names[top1_idx]

        # Draw top prediction
        label = f"Prediction: {class_name} ({top1_conf:.1%})"
        color = (0, 255, 0) if class_name != "no_drone" else (0, 0, 255)
        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # Draw ALL class probabilities as a bar chart (using smoothed values)
        y_start = 80
        for i in range(num_classes):
            cls_name = class_names[i]
            prob = avg_probs[i]
            bar_width = int(prob * 300)
            bar_color = (0, 255, 0) if i == top1_idx else (200, 200, 200)
            y = y_start + i * 35

            # Class name and percentage
            text = f"{cls_name}: {prob:.1%}"
            cv2.putText(frame, text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            # Probability bar
            cv2.rectangle(frame, (220, y - 15), (220 + bar_width, y + 5), bar_color, -1)

        # Drone detected banner
        if class_name != "no_drone":
            cv2.putText(frame, "DRONE DETECTED", (20, y_start + 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        # Frame count indicator
        filled = len(prob_history)
        cv2.putText(frame, f"Buffer: {filled}/{SMOOTHING_FRAMES}", (20, y_start + 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("Drone Classification - Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Goodbye!")

if __name__ == "__main__":
    main()
