"""
Layer 1 PPE Detection – Final Optimized Version
- ONNX model (best.onnx) for speed
- Lightweight IOU tracker (no ultralytics tracking overhead)
- Frame resizing + optional skipping
- FPS counter
- Vest compliance checks (optional, disabled by default for max FPS)
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

# ============================================================
# CONFIGURATION – Adjust these to your preference
# ============================================================
MODEL_PATH = "models/weights/best.onnx"       # ONNX model (fast)
VIDEO_PATH = "miners.mp4"                     # Video file (in same folder)
CONF_THRESH = 0.5
IMGSZ = 320                                   # Model input size
FRAME_SKIP = 1                                # Process every frame (1) – can increase to 2 or 3
PRE_RESIZE = (640, 480)                       # Resize frame before inference (smaller = faster)
ENABLE_TRACKING = True                        # Set False to disable tracking (even faster)
ENABLE_VEST_CHECKS = False                    # Set True to enable colour/stripe checks (slower)
SHOW_FPS = True

# IOU threshold for tracker (higher = stricter)
IOU_THRESH = 0.4

# HSV ranges for vest checks (only used if ENABLE_VEST_CHECKS = True)
GREEN_LOWER = np.array([30, 100, 100])
GREEN_UPPER = np.array([70, 255, 255])
ORANGE_LOWER = np.array([5, 100, 100])
ORANGE_UPPER = np.array([15, 255, 255])
REFLECTIVE_THRESH = 200
MIN_AREA = 100
MIN_ASPECT_RATIO = 3

# Force ONNX to use multiple threads
os.environ["OMP_NUM_THREADS"] = "4"

# ============================================================
# LIGHTWEIGHT IOU TRACKER (replaces ultralytics track)
# ============================================================
def iou(box1, box2):
    """Intersection over Union between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

class LightweightTracker:
    def __init__(self, iou_thresh=0.4, max_age=5):
        self.next_id = 1
        self.tracks = []      # each track: [id, box, age]
        self.iou_thresh = iou_thresh
        self.max_age = max_age

    def update(self, detections):
        """detections: list of [x1,y1,x2,y2] boxes"""
        matched = set()
        new_tracks = []
        # Match detections to existing tracks
        for det in detections:
            best_id = None
            best_iou = -1
            for i, (tid, tbox, age) in enumerate(self.tracks):
                if i in matched:
                    continue
                cur_iou = iou(det, tbox)
                if cur_iou > best_iou and cur_iou > self.iou_thresh:
                    best_iou = cur_iou
                    best_id = i
            if best_id is not None:
                matched.add(best_id)
                new_tracks.append((self.tracks[best_id][0], det, 0))
            else:
                # New detection – assign new ID
                new_tracks.append((self.next_id, det, 0))
                self.next_id += 1
        # Keep unmatched tracks for max_age frames
        for i, (tid, tbox, age) in enumerate(self.tracks):
            if i not in matched and age < self.max_age:
                new_tracks.append((tid, tbox, age + 1))
        self.tracks = new_tracks
        # Return mapping from detection index to track ID (for boxes in same order as input)
        # Build dict: box tuple -> track id (approximate)
        box_to_id = {}
        for tid, tbox, _ in self.tracks:
            # Use bounding box center as key (simple)
            key = (int((tbox[0]+tbox[2])/2), int((tbox[1]+tbox[3])/2))
            box_to_id[tuple(tbox)] = tid
        ids = []
        for det in detections:
            # find track with matching box
            found = False
            for tid, tbox, _ in self.tracks:
                if iou(det, tbox) > 0.5:
                    ids.append(tid)
                    found = True
                    break
            if not found:
                ids.append(None)
        return ids

# ============================================================
# VEST CHECK FUNCTIONS (only if enabled)
# ============================================================
def check_color_compliance(roi):
    if roi is None or roi.size == 0:
        return False, "empty"
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    orange_mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
    total = roi.size / 3
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

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    model = YOLO(MODEL_PATH)
    print("Loaded ONNX model.")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video source.")
        return

    # Initialize lightweight tracker if enabled
    tracker = LightweightTracker(iou_thresh=IOU_THRESH) if ENABLE_TRACKING else None

    frame_count = 0
    fps_timer = time.time()
    fps_counter = 0
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Resize frame for faster inference
        small_frame = cv2.resize(frame, PRE_RESIZE)

        # Run inference every FRAME_SKIP frames
        if frame_count % FRAME_SKIP == 0:
            results = model(small_frame, imgsz=IMGSZ, conf=CONF_THRESH, verbose=False)
            detections = results[0].boxes

            # Scale bounding boxes back to original frame size
            h_orig, w_orig = frame.shape[:2]
            h_small, w_small = small_frame.shape[:2]
            x_scale = w_orig / w_small
            y_scale = h_orig / h_small

            boxes = []
            classes = []
            confs = []
            if detections is not None:
                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Scale coordinates
                    x1 = int(x1 * x_scale)
                    y1 = int(y1 * y_scale)
                    x2 = int(x2 * x_scale)
                    y2 = int(y2 * y_scale)
                    boxes.append([x1, y1, x2, y2])
                    classes.append(model.names[int(box.cls[0])])
                    confs.append(float(box.conf[0]))

            # Apply tracking to assign IDs
            if ENABLE_TRACKING and boxes:
                track_ids = tracker.update(boxes)   # returns list of IDs matching boxes order
            else:
                track_ids = [None] * len(boxes)

            # Annotate frame
            annotated = frame.copy()
            for i, (box, cls, conf, tid) in enumerate(zip(boxes, classes, confs, track_ids)):
                x1, y1, x2, y2 = box
                color = (0, 255, 0) if cls == "vest" else (0, 255, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{'ID:'+str(tid)+' ' if tid else ''}{cls}: {conf:.2f}"
                cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Optional vest checks
                if ENABLE_VEST_CHECKS and cls == "vest":
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        roi_small = cv2.resize(roi, (0,0), fx=0.5, fy=0.5)
                        compliant, col_name = check_color_compliance(roi_small)
                        stripes = has_reflective_stripes(roi_small)
                        info = f"Color:{col_name} Stripes:{stripes}"
                        cv2.putText(annotated, info, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        if not compliant or not stripes:
                            print(f"VEST VIOLATION - ID:{tid} - Color:{col_name}, Stripes:{stripes}")

            last_annotated = annotated
            fps_counter += 1
        else:
            # On skipped frames, reuse last annotated frame
            last_annotated = last_annotated if 'last_annotated' in locals() else frame

        # FPS calculation
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        display_frame = last_annotated if 'last_annotated' in locals() else frame
        if SHOW_FPS:
            cv2.putText(display_frame, f"FPS: {fps_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Resize display for faster rendering (optional)
        small_display = cv2.resize(display_frame, (960, 540))
        cv2.imshow("PPE Detection (Optimized)", small_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()