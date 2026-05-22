import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
from vlm_async import AsyncVlmAnalyzer

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
MODEL_PATH = "models/weights/best.onnx"
VIDEO_PATH = "D:\\Layer 1\\jindal-ppe-detection\\15561206_1920_1080_25fps.mp4"
CONF_THRESH = 0.25
IMGSZ = 320
FRAME_SKIP = 1
PRE_RESIZE = (640, 480)
ENABLE_TRACKING = True
ENABLE_VEST_CHECKS = False
ENABLE_DEEP_ANALYSIS = True
VLM_ANALYSIS_INTERVAL = 300          # 10 seconds between analyses for same worker
IOU_THRESH = 0.4
SHOW_FPS = True
MIN_BOX_AREA = 5000                  # pixels – ignore very far/small persons

# Vest colour ranges (unused if ENABLE_VEST_CHECKS=False)
GREEN_LOWER = np.array([30, 100, 100])
GREEN_UPPER = np.array([70, 255, 255])
ORANGE_LOWER = np.array([5, 100, 100])
ORANGE_UPPER = np.array([15, 255, 255])
REFLECTIVE_THRESH = 200
MIN_AREA = 100
MIN_ASPECT_RATIO = 3

os.environ["OMP_NUM_THREADS"] = "4"

# ------------------------------------------------------------
# CLEANUP FUNCTION (removes old snapshots on startup)
# ------------------------------------------------------------
def cleanup_old_snapshots(max_keep=100):
    snap_dir = "snapshots"
    if not os.path.exists(snap_dir):
        return
    files = [os.path.join(snap_dir, f) for f in os.listdir(snap_dir) if f.endswith('.jpg')]
    files.sort(key=os.path.getmtime)
    deleted = 0
    while len(files) > max_keep:
        os.remove(files.pop(0))
        deleted += 1
    if deleted:
        print(f"Cleaned up {deleted} old snapshots, remaining {len(files)}.")

# ------------------------------------------------------------
# LIGHTWEIGHT IOU TRACKER (same as before)
# ------------------------------------------------------------
def iou(box1, box2):
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
        self.tracks = []
        self.iou_thresh = iou_thresh
        self.max_age = max_age

    def update(self, detections):
        matched = set()
        new_tracks = []
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
                new_tracks.append((self.next_id, det, 0))
                self.next_id += 1
        for i, (tid, tbox, age) in enumerate(self.tracks):
            if i not in matched and age < self.max_age:
                new_tracks.append((tid, tbox, age + 1))
        self.tracks = new_tracks
        ids = []
        for det in detections:
            found = False
            for tid, tbox, _ in self.tracks:
                if iou(det, tbox) > 0.5:
                    ids.append(tid)
                    found = True
                    break
            if not found:
                ids.append(None)
        return ids

# ------------------------------------------------------------
# VEST CHECKS (optional, keep for completeness)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
def main():
    cleanup_old_snapshots()

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    model = YOLO(MODEL_PATH)
    print("Loaded YOLO ONNX model.")
    print(f"Model classes: {model.names}")

    vlm = None
    worker_last_analyzed = {}
    if ENABLE_DEEP_ANALYSIS:
        vlm = AsyncVlmAnalyzer()
        print("Deep analysis enabled (only when missing PPE).")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Cannot open video source.")
        return

    tracker = LightweightTracker(iou_thresh=IOU_THRESH) if ENABLE_TRACKING else None

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
        small_frame = cv2.resize(frame, PRE_RESIZE)

        if frame_count % FRAME_SKIP == 0:
            results = model(small_frame, imgsz=IMGSZ, conf=CONF_THRESH, verbose=False)
            detections = results[0].boxes

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
                    boxes.append([int(x1*x_scale), int(y1*y_scale), int(x2*x_scale), int(y2*y_scale)])
                    classes.append(model.names[int(box.cls[0])])
                    confs.append(float(box.conf[0]))

            if ENABLE_TRACKING and boxes:
                track_ids = tracker.update(boxes)
            else:
                track_ids = [None] * len(boxes)

            # ----- Build a map of missing PPE per worker -----
            missing_ppe_per_worker = {}
            for i, (cls, tid) in enumerate(zip(classes, track_ids)):
                if tid is None:
                    continue
                if cls in ["no_helmet", "no_vest"]:
                    missing_ppe_per_worker.setdefault(tid, []).append(cls)

            # ----- Annotate and trigger VLM only for persons with missing PPE -----
            annotated = frame.copy()
            for i, (box, cls, conf, tid) in enumerate(zip(boxes, classes, confs, track_ids)):
                x1, y1, x2, y2 = box
                color = (0, 255, 0) if cls == "vest" else (0, 255, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{f'ID:{tid} ' if tid else ''}{cls}: {conf:.2f}"
                cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if ENABLE_VEST_CHECKS and cls == "vest":
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        roi_small = cv2.resize(roi, (0,0), fx=0.5, fy=0.5)
                        compliant, col_name = check_color_compliance(roi_small)
                        stripes = has_reflective_stripes(roi_small)
                        info = f"Color:{col_name} Stripes:{stripes}"
                        cv2.putText(annotated, info, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # ----- VLM trigger: only person, with missing PPE, and box area > threshold -----
                if vlm and cls == "person" and tid is not None:
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area < MIN_BOX_AREA:
                        continue   # too far / too small
                    if tid in missing_ppe_per_worker:   # has missing PPE
                        cur_frame = frame_count
                        last = worker_last_analyzed.get(tid, 0)
                        if cur_frame - last >= VLM_ANALYSIS_INTERVAL:
                            worker_last_analyzed[tid] = cur_frame
                            # Ensure valid ROI
                            y1 = max(0, min(y1, frame.shape[0]-1))
                            y2 = max(y1+1, min(y2, frame.shape[0]))
                            x1 = max(0, min(x1, frame.shape[1]-1))
                            x2 = max(x1+1, min(x2, frame.shape[1]))
                            if y2 > y1 and x2 > x1:
                                os.makedirs("snapshots", exist_ok=True)
                                snap_path = f"snapshots/worker_{tid}_{cur_frame}.jpg"
                                success = cv2.imwrite(snap_path, frame[y1:y2, x1:x2])
                                print(f"Snapshot saved: {snap_path} -> success={success}")
                                if success:
                                    vlm.analyze_async(snap_path, tid)
                            else:
                                print(f"Invalid ROI for worker {tid}: {x1},{y1},{x2},{y2}")

            last_annotated = annotated
            fps_counter += 1
        else:
            last_annotated = last_annotated if last_annotated is not None else frame

        # FPS display
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        display_frame = last_annotated if last_annotated is not None else frame
        if SHOW_FPS:
            cv2.putText(display_frame, f"FPS: {fps_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        small_display = cv2.resize(display_frame, (960, 540))
        cv2.imshow("PPE Detection + Deep Analysis", small_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if vlm:
        vlm.shutdown()

if __name__ == "__main__":
    main()