# benchmark_openvino.py
import cv2
import time
from ultralytics import YOLO

model = YOLO("models/weights/best_openvino_model/")
cap = cv2.VideoCapture(r"D:\Layer 1\jindal-ppe-detection\miners.mp4")

start = time.time()
frame_count = 0
while frame_count < 100:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, imgsz=320, conf=0.5, verbose=False)
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames...")
elapsed = time.time() - start
print(f"Average inference time: {elapsed/frame_count*1000:.1f} ms")
print(f"Max FPS: {frame_count/elapsed:.1f}")
cap.release()