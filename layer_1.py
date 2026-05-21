import cv2
import time
from ultralytics import YOLO
import os

VIDEO_PATH = r"D:\Layer 1\jindal-ppe-detection\miners.mp4"
MODEL_PATH = "models/weights/best.onnx"
CONF_THRESH = 0.5
IMGSZ = 320
PRE_RESIZE = (640, 480)   # resize frame before inference

os.environ["OMP_NUM_THREADS"] = "4"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

fps_counter = 0
fps_timer = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster inference
    small = cv2.resize(frame, PRE_RESIZE)
    
    # Run detection (no tracking)
    results = model(small, imgsz=IMGSZ, conf=CONF_THRESH, verbose=False)
    
    # Draw boxes (optional – comment out to measure pure speed)
    annotated = results[0].plot()
    
    # FPS counter
    fps_counter += 1
    if time.time() - fps_timer >= 1.0:
        print(f"FPS: {fps_counter}")
        fps_counter = 0
        fps_timer = time.time()
    
    # Show (resize display to smaller window for speed)
    display = cv2.resize(annotated, (960, 540))
    cv2.imshow("Minimal", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()