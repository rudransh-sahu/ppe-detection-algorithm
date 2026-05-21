"""
Layer 1 PPE Detection – ONNX + Tracking
No risk zones – just detection, tracking, and vest compliance.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
MODEL_PATH = "models/weights/best.onnx"   # ONNX model (or fallback to .pt)
CONF_THRESH = 0.5
IMGSZ = 416
FRAME_SKIP = 2                            # Process every 2nd frame
SOURCE = 0                                # 0 = webcam, or "video.mp4"
SHOW_FPS = True

# HSV ranges for vest colour (calibrate for your site)
GREEN_LOWER = np.array([30, 100, 100])
GREEN_UPPER = np.array([70, 255, 255])
ORANGE_LOWER = np.array([5, 100, 100])
ORANGE_UPPER = np.array([15, 255, 255])

# Reflective stripe detection
REFLECTIVE_THRESH = 200
MIN_AREA = 100
MIN_ASPECT_RATIO = 3

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def check_color_compliance(roi):
    """Return (is_compliant, color_name)."""
    if roi is None or roi.size == 0:
        return False, "empty"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    orange_mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
    total = roi.size / 3
    if total == 0:
        return False, "zero"
    green_ratio = cv2.countNonZero(green_mask) / total
    orange_ratio = cv2.countNonZero(orange_mask) / total
    if green_ratio > 0.2:
        return True, "green"
    if orange_ratio > 0.2:
        return True, "orange"
    return False, "none"

def has_reflective_stripes(roi):
    """Return True if reflective stripes are present."""
    if roi is None or roi.size == 0:
        return False
    _, _, v = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    _, thresh = cv2.threshold(v, REFLECTIVE_THRESH, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h != 0 else 0
        if aspect > MIN_ASPECT_RATIO:
            return True
    return False

# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def main():
    # Load model – prefer ONNX
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print("Loaded ONNX model.")
    else:
        pt_path = "models/weights/best.pt"
        if os.path.exists(pt_path):
            model = YOLO(pt_path)
            print("Loaded PyTorch model. Export to ONNX for better performance.")
        else:
            print("ERROR: No model found. Place best.pt in models/weights/")
            return

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print("Cannot open video source.")
        return

    frame_count = 0
    last_annotated = None
    fps_timer = time.time()
    fps_counter = 0
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Frame skipping
        if frame_count % FRAME_SKIP == 0:
            # Run tracking
            results = model.track(frame, persist=True, conf=CONF_THRESH, imgsz=IMGSZ)
            detections = results[0].boxes

            # Get track IDs if available
            if detections is not None and detections.id is not None:
                track_ids = detections.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(detections) if detections is not None else []

            annotated = frame.copy()

            if detections is not None:
                for i, box in enumerate(detections):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    track_id = track_ids[i] if i < len(track_ids) else "?"

                    # Box colour: green for vest, yellow for others
                    color = (0, 255, 0) if cls_name == "vest" else (0, 255, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{track_id} {cls_name}: {conf:.2f}"
                    cv2.putText(annotated, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Additional vest compliance checks
                    if cls_name == "vest":
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            roi_small = cv2.resize(roi, (0,0), fx=0.5, fy=0.5)
                            compliant, col_name = check_color_compliance(roi_small)
                            stripes = has_reflective_stripes(roi_small)
                            info = f"Color:{col_name} Stripes:{stripes}"
                            cv2.putText(annotated, info, (x1, y2+15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                            if not compliant or not stripes:
                                print(f"VEST VIOLATION - Worker ID:{track_id} - Color:{col_name}, Stripes:{stripes}")

            last_annotated = annotated
            fps_counter += 1

        # FPS calculation
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        # Display
        display_frame = last_annotated if last_annotated is not None else frame
        if SHOW_FPS:
            cv2.putText(display_frame, f"FPS: {fps_display}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("PPE Detection (ONNX)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()