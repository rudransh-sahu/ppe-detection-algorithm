"""
Layer 2 – Risk Zone Monitoring with PPE Tracking
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import yaml

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
MODEL_PATH = "models/weights/best.onnx"  
CONF_THRESH = 0.5
IMGSZ = 416
FRAME_SKIP = 2
SOURCE = r"D:\miners.mp4"
SHOW_FPS = True

ZONES_CONFIG = "config/risk_zones.yaml"

# HSV ranges (same as before)
GREEN_LOWER = np.array([30, 100, 100])
GREEN_UPPER = np.array([70, 255, 255])
ORANGE_LOWER = np.array([5, 100, 100])
ORANGE_UPPER = np.array([15, 255, 255])
REFLECTIVE_THRESH = 200
MIN_AREA = 100
MIN_ASPECT_RATIO = 3

# ------------------------------------------------------------
# Helper functions (vest checks)
# ------------------------------------------------------------
def check_color_compliance(roi):
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
# Zone management
# ------------------------------------------------------------
def load_zones(config_path):
    if not os.path.exists(config_path):
        print(f"Warning: {config_path} not found. Creating default empty file.")
        # Create directory if needed
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            f.write("zones: []\n")
        return []
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if data is None:
        print(f"Warning: {config_path} is empty. Using empty zones list.")
        return []
    
    zones = []
    for z in data.get('zones', []):
        zones.append({
            'name': z['name'],
            'polygon': np.array(z['vertices'], dtype=np.int32),
            'required_ppe': set(z['required_ppe']),
            'trigger_vlm': z.get('trigger_vlm', False)
        })
    return zones

def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: ONNX model not found at {MODEL_PATH}")
        return
    model = YOLO(MODEL_PATH)
    print("Loaded ONNX model with tracking.")

    # Load risk zones
    if not os.path.exists(ZONES_CONFIG):
        print(f"ERROR: Zones config not found at {ZONES_CONFIG}")
        return
    zones = load_zones(ZONES_CONFIG)
    print(f"Loaded {len(zones)} risk zones.")

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print("Cannot open video source.")
        return

    frame_count = 0
    last_annotated = None
    fps_timer = time.time()
    fps_counter = 0
    fps_display = 0

    # To avoid repeated alerts for the same (zone, worker_id) pair
    alerted = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_SKIP == 0:
            results = model.track(frame, persist=True, conf=CONF_THRESH, imgsz=IMGSZ)
            detections = results[0].boxes

            if detections is not None:
                if detections.id is not None:
                    track_ids = detections.id.int().cpu().tolist()
                else:
                    track_ids = [None] * len(detections)

                annotated = frame.copy()

                for i, box in enumerate(detections):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    track_id = track_ids[i] if i < len(track_ids) else None

                    # Draw box
                    color = (0, 255, 0) if cls_name == "vest" else (0, 255, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{track_id} {cls_name}: {conf:.2f}"
                    cv2.putText(annotated, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Only consider 'person' for zone entry (tracking only for workers)
                    if cls_name == "person" and track_id is not None:
                        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
                        # Check each zone
                        for zone in zones:
                            if point_in_polygon(centroid, zone['polygon']):
                                # Draw zone outline on annotated frame
                                cv2.polylines(annotated, [zone['polygon']], True, (0, 0, 255), 2)
                                cv2.putText(annotated, zone['name'], (zone['polygon'][0][0], zone['polygon'][0][1]-5),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                                # Check required PPE
                                worker_has = set()
                                # We need to collect all PPE detected for this worker in current frame
                                # Simple approach: store in a dict keyed by track_id
                                # For now, we'll simulate by looking at other detections with same track_id
                                # But since we are inside per-detection loop, we can accumulate later.
                                # Let's refactor: first gather all detections by track_id, then check.
                                # We'll implement that elegantly after this loop.
                                pass

                # ---------- Second pass: gather PPE per track_id and evaluate zones ----------
                # Gather all objects by track_id
                objects_by_id = {}
                for i, box in enumerate(detections):
                    if detections.id is not None:
                        tid = track_ids[i] if i < len(track_ids) else None
                        if tid is None:
                            continue
                        if tid not in objects_by_id:
                            objects_by_id[tid] = {'bbox': None, 'classes': set(), 'centroid': None}
                        cls_name = model.names[int(box.cls[0])]
                        objects_by_id[tid]['classes'].add(cls_name)
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        objects_by_id[tid]['centroid'] = ((x1+x2)//2, (y1+y2)//2)
                        objects_by_id[tid]['bbox'] = (x1,y1,x2,y2)

                # Now evaluate each worker in each zone
                for tid, data in objects_by_id.items():
                    cent = data['centroid']
                    if cent is None:
                        continue
                    for zone in zones:
                        if point_in_polygon(cent, zone['polygon']):
                            # Check if required PPE is missing
                            missing = zone['required_ppe'] - data['classes']
                            if missing:
                                alert_key = f"{tid}_{zone['name']}"
                                if alert_key not in alerted:
                                    print(f"⚠️ ALERT: Worker {tid} entered {zone['name']} missing: {missing}")
                                    # Optionally save snapshot
                                    x1,y1,x2,y2 = data['bbox']
                                    roi = frame[y1:y2, x1:x2]
                                    if roi.size > 0:
                                        cv2.imwrite(f"alerts/alert_{tid}_{int(time.time())}.jpg", roi)
                                    # Trigger VLM placeholder
                                    if zone['trigger_vlm']:
                                        print(f"   → Triggering VLM analysis for worker {tid}")
                                        # Here you would call your VLM (e.g., Qwen2.5-VL)
                                    alerted.add(alert_key)
                            else:
                                # Worker is compliant, reset alert flag if they leave later
                                # For simplicity, we remove from alerted when they exit zone (handled by not being inside)
                                pass

                # Reset alerted for workers no longer in any zone? We can do periodic cleanup.
                # For now, it will stop printing once they become compliant.

                last_annotated = annotated
                fps_counter += 1

        # FPS and display (same as before)
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        display_frame = last_annotated if last_annotated is not None else frame
        if SHOW_FPS:
            cv2.putText(display_frame, f"FPS: {fps_display}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Risk Zone Monitoring", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()