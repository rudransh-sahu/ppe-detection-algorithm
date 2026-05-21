"""
Layer 1 PPE Detection – Real‑time (30 FPS)
- No frame skipping (process every frame)
- Lightweight tracking (ByteTrack built into model.track)
- Optional vest checks disabled for max speed
- Display resized for faster rendering
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

# ============================================================
# CONFIGURATION
# ============================================================
VIDEO_PATH = "D:\Layer 1\jindal-ppe-detection\miners.mp4"
MODEL_PATH = "models/weights/best.onnx"       # or OpenVINO folder
CONF_THRESH = 0.5
IMGSZ = 320                                   # matches your export
FRAME_SKIP = 1                                # process EVERY frame
ENABLE_VEST_CHECKS = True                # disable for speed
SHOW_FPS = True

# Resize frame before inference (makes pre‑processing faster)
PRE_RESIZE_WIDTH = 640
PRE_RESIZE_HEIGHT = 480

# Force ONNX to use multiple threads
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["ONNXRUNTIME_EXECUTION_PROVIDER"] = "CPUExecutionProvider"

# ============================================================
# Helper functions (only if ENABLE_VEST_CHECKS is True)
# ============================================================
def check_color_compliance(roi):
    # ... (same as before, omitted for brevity)
    return False, "none"

def has_reflective_stripes(roi):
    return False

# ============================================================
# Model loading
# ============================================================
def load_model():
    if os.path.exists("models/weights/best_openvino_model/"):
        return YOLO("models/weights/best_openvino_model/")
    return YOLO(MODEL_PATH)

# ============================================================
# Main pipeline
# ============================================================
def main():
    model = load_model()
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video.")
        return

    frame_count = 0
    fps_timer = time.time()
    fps_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Resize for faster inference
        small_frame = cv2.resize(frame, (PRE_RESIZE_WIDTH, PRE_RESIZE_HEIGHT))

        # Run tracking (no frame skip)
        results = model.track(small_frame, persist=True, conf=CONF_THRESH, imgsz=IMGSZ, verbose=False)
        detections = results[0].boxes

        # Scale coordinates back to original frame size
        h_orig, w_orig = frame.shape[:2]
        h_small, w_small = small_frame.shape[:2]
        x_scale = w_orig / w_small
        y_scale = h_orig / h_small

        annotated = frame.copy()

        if detections is not None:
            track_ids = detections.id.int().cpu().tolist() if detections.id is not None else [None]*len(detections)
            for i, box in enumerate(detections):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = int(x1 * x_scale)
                y1 = int(y1 * y_scale)
                x2 = int(x2 * x_scale)
                y2 = int(y2 * y_scale)
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                track_id = track_ids[i] if i < len(track_ids) else "?"

                color = (0, 255, 0) if cls_name == "vest" else (0, 255, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{track_id} {cls_name}: {conf:.2f}"
                cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # FPS counter
        fps_counter += 1
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        if SHOW_FPS:
            cv2.putText(annotated, f"FPS: {fps_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize display for faster rendering (optional)
        display = cv2.resize(annotated, (960, 540))
        cv2.imshow("PPE Detection (Real-time)", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()