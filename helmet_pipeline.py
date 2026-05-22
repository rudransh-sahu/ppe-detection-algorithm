import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import uuid
from helmet_classifier import HelmetClassifier

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
MODEL_PATH = "models/weights/best.onnx"
VIDEO_PATH = "complete.mp4"
CONF_THRESH = 0.25
IMGSZ = 320
FRAME_SKIP = 1
PRE_RESIZE = (640, 480)
ENABLE_TRACKING = True
ENABLE_VEST_CHECKS = False
ENABLE_HELMET_CLASSIFIER = True          # <-- new flag
IOU_THRESH = 0.4
SHOW_FPS = True
MIN_BOX_AREA = 5000
TEMPORAL_VOTE_WINDOW = 5
TEMPORAL_VOTE_THRESHOLD = 1               # any missing PPE triggers alert (testing)

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
# CLEANUP & INIT
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
# LIGHTWEIGHT IOU TRACKER (ONLY FOR PERSON CLASS)
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
                if iou(det, tbox) > self.iou_thresh:
                    ids.append(tid)
                    found = True
                    break
            if not found:
                ids.append(None)
        return ids

# ------------------------------------------------------------
# VEST CHECKS (optional)
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
# TEMPORAL SMOOTHING
# ------------------------------------------------------------
class TemporalVote:
    def __init__(self, window_size=TEMPORAL_VOTE_WINDOW, threshold=TEMPORAL_VOTE_THRESHOLD):
        self.window = []
        self.window_size = window_size
        self.threshold = threshold
    def add(self, value):
        self.window.append(value)
        if len(self.window) > self.window_size:
            self.window.pop(0)
    def is_triggered(self):
        if len(self.window) < self.window_size:
            return False
        return sum(self.window) >= self.threshold

# ------------------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------------------
def main():
    os.makedirs("snapshots", exist_ok=True)
    cleanup_old_snapshots()

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return
    model = YOLO(MODEL_PATH)
    class_names = model.names
    PERSON_CLASS = "Person"
    MISSING_PPE_CLASSES = ["no_helmet", "no_goggle", "no_gloves", "no_boots"]
    VEST_CLASS = "vest"
    print("Loaded YOLO ONNX model.")
    print(f"Model classes: {class_names}")

    # --------------------------------------------------------
    # Helmet classifier (zero‑shot VLM)
    # --------------------------------------------------------
    helmet_clf = None
    if ENABLE_HELMET_CLASSIFIER:
        helmet_clf = HelmetClassifier()
        print("Zero‑shot helmet classifier enabled (background).")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Cannot open video source: {VIDEO_PATH}")
        return

    tracker = LightweightTracker(iou_thresh=IOU_THRESH) if ENABLE_TRACKING else None

    frame_count = 0
    last_annotated = None
    fps_timer = time.time()
    fps_counter = 0
    fps_display = 0

    # Font settings – larger for legibility
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 3

    # Vote storage for alert (only uses YOLO's no_helmet)
    worker_vote = {}

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

            person_boxes = []
            person_confs = []
            equipment_boxes = []
            equipment_classes = []
            all_detections = []

            if detections is not None:
                for box in detections:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = class_names[int(box.cls[0])]
                    scaled_box = [int(x1*x_scale), int(y1*y_scale), int(x2*x_scale), int(y2*y_scale)]
                    conf = float(box.conf[0])
                    all_detections.append((scaled_box, cls, conf))
                    if cls == PERSON_CLASS:
                        person_boxes.append(scaled_box)
                        person_confs.append(conf)
                    elif cls in MISSING_PPE_CLASSES:
                        equipment_boxes.append(scaled_box)
                        equipment_classes.append(cls)

            # Track only persons
            if ENABLE_TRACKING and person_boxes:
                track_ids = tracker.update(person_boxes)
            else:
                track_ids = [None] * len(person_boxes)

            # Build missing PPE per person via IoU (low threshold)
            missing_ppe_per_worker = {}
            for i, pbox in enumerate(person_boxes):
                tid = track_ids[i] if i < len(track_ids) else None
                if tid is None:
                    continue
                for ebox, ecls in zip(equipment_boxes, equipment_classes):
                    if iou(pbox, ebox) > 0.15:
                        missing_ppe_per_worker.setdefault(tid, []).append(ecls)

            # Update temporal votes (for alert – based on no_helmet)
            for tid in set(track_ids):
                if tid is None:
                    continue
                if tid not in worker_vote:
                    worker_vote[tid] = TemporalVote()
                has_missing = 1 if tid in missing_ppe_per_worker else 0
                worker_vote[tid].add(has_missing)

            # Draw all detections (equipment + persons)
            annotated = frame.copy()
            for box, cls, conf in all_detections:
                x1, y1, x2, y2 = box
                if cls == VEST_CLASS:
                    color = (0, 255, 0)
                elif cls in MISSING_PPE_CLASSES:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{cls}: {conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(label, font, 0.7, 2)
                text_x = x1
                text_y = y1 - 5
                if text_y - th < 0:
                    text_y = y1 + th + 5
                cv2.rectangle(annotated, (text_x, text_y - th - baseline), (text_x + tw, text_y + baseline), (0,0,0), -1)
                cv2.putText(annotated, label, (text_x, text_y), font, 0.7, (255,255,255), 2)

            # Alert: thick red box + "MISSING HELMET" for YOLO‑detected violation
            for i, pbox in enumerate(person_boxes):
                tid = track_ids[i] if i < len(track_ids) else None
                if tid is None:
                    continue
                if tid in worker_vote and worker_vote[tid].is_triggered():
                    x1, y1, x2, y2 = pbox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    alert_text = "MISSING HELMET"
                    (tw, th), baseline = cv2.getTextSize(alert_text, font, 1.2, 3)
                    text_x = x1
                    text_y = y1 - 10
                    if text_y - th < 0:
                        text_y = y1 + th + 10
                    cv2.rectangle(annotated, (text_x, text_y - th - baseline), (text_x + tw, text_y + baseline), (0,0,255), -1)
                    cv2.putText(annotated, alert_text, (text_x, text_y), font, 1.2, (255,255,255), 3)

            # Draw track IDs
            for i, pbox in enumerate(person_boxes):
                tid = track_ids[i] if i < len(track_ids) else None
                if tid is not None:
                    x1, y1, x2, y2 = pbox
                    id_text = f"ID:{tid}"
                    (tw, th), baseline = cv2.getTextSize(id_text, font, 0.8, 2)
                    cv2.rectangle(annotated, (x1, y1-25), (x1+tw, y1-5), (0,0,0), -1)
                    cv2.putText(annotated, id_text, (x1, y1-10), font, 0.8, (255,255,0), 2)

            # ----------------------------------------------------
            # Helmet classifier (zero‑shot VLM) trigger
            # ----------------------------------------------------
            if helmet_clf:
                for i, pbox in enumerate(person_boxes):
                    tid = track_ids[i] if i < len(track_ids) else None
                    if tid is None:
                        continue
                    # Crop head region (upper half of person box)
                    x1, y1, x2, y2 = pbox
                    head_y2 = y1 + int((y2 - y1) * 0.4)   # top 40% of the person
                    head_y1 = max(0, y1 - int((y2 - y1) * 0.2))  # a little above head
                    head_x1 = max(0, x1)
                    head_x2 = min(frame.shape[1], x2)
                    if head_y2 > head_y1 and head_x2 > head_x1:
                        head_roi = frame[head_y1:head_y2, head_x1:head_x2]
                        if head_roi.size > 0:
                            temp_path = f"temp_head_{tid}_{frame_count}.jpg"
                            cv2.imwrite(temp_path, head_roi)
                            request_id = str(uuid.uuid4())[:8]
                            helmet_clf.classify_async(temp_path, frame_count, request_id)

            last_annotated = annotated.copy()
            fps_counter += 1

        # On skipped frames, reuse last annotated
        display_frame = last_annotated if last_annotated is not None else frame

        # FPS display
        if time.time() - fps_timer >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            fps_timer = time.time()

        if SHOW_FPS:
            display_copy = display_frame.copy()
            fps_label = f"FPS: {fps_display}"
            (fw, fh), baseline = cv2.getTextSize(fps_label, font, 0.9, 2)
            cv2.rectangle(display_copy, (5, 5), (5+fw+5, 5+fh+5), (0,0,0), -1)
            cv2.putText(display_copy, fps_label, (10, 10+fh), font, 0.9, (0,255,0), 2)
            display_frame = display_copy

        small_display = cv2.resize(display_frame, (960, 540))
        cv2.imshow("PPE Detection + Helmet Classifier", small_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if helmet_clf:
        helmet_clf.shutdown()

if __name__ == "__main__":
    main()