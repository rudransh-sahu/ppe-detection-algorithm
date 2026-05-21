"""
Benchmark YOLO model inference speed on a video file.
Measures average inference time per frame and maximum achievable FPS.
"""

import cv2
import time
from ultralytics import YOLO
import os

# ============================================================
# CONFIGURATION
# ============================================================
VIDEO_PATH = r"D:\miners.mp4"
MODEL_PATH = "models/weights/best.onnx"   # or "models/weights/best_openvino_model/"
CONF_THRESH = 0.5
IMGSZ = 320                               # Must match model export size
NUM_FRAMES_TO_BENCHMARK = 200             # Number of frames to process (adjust as needed)

# ============================================================
# Load model
# ============================================================
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    exit(1)

model = YOLO(MODEL_PATH)
print(f"Loaded model: {MODEL_PATH}")

# ============================================================
# Open video
# ============================================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open video {VIDEO_PATH}")
    exit(1)

fps_video = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps_video:.2f}")

# ============================================================
# Benchmark loop
# ============================================================
frame_count = 0
total_inference_time = 0.0
inference_times = []

print(f"\nBenchmarking {NUM_FRAMES_TO_BENCHMARK} frames...\n")

while frame_count < NUM_FRAMES_TO_BENCHMARK:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.perf_counter()
    results = model(frame, imgsz=IMGSZ, conf=CONF_THRESH, verbose=False)
    elapsed = time.perf_counter() - start

    total_inference_time += elapsed
    inference_times.append(elapsed)
    frame_count += 1

    # Optional: print progress every 10 frames
    if frame_count % 10 == 0:
        print(f"  Processed {frame_count} frames...")

cap.release()

# ============================================================
# Calculate statistics
# ============================================================
if frame_count == 0:
    print("No frames processed.")
    exit(1)

avg_time_ms = (total_inference_time / frame_count) * 1000
avg_fps = 1.0 / (total_inference_time / frame_count)

# Calculate percentiles for stability
sorted_times = sorted(inference_times)
p50_ms = sorted_times[int(len(sorted_times)*0.5)] * 1000
p95_ms = sorted_times[int(len(sorted_times)*0.95)] * 1000
p99_ms = sorted_times[int(len(sorted_times)*0.99)] * 1000

print("\n" + "="*50)
print("BENCHMARK RESULTS")
print("="*50)
print(f"Frames processed:     {frame_count}")
print(f"Model:                {os.path.basename(MODEL_PATH)}")
print(f"Image size:           {IMGSZ}x{IMGSZ}")
print(f"Confidence threshold: {CONF_THRESH}")
print("\nInference Times:")
print(f"  Average:            {avg_time_ms:.1f} ms")
print(f"  Median (p50):       {p50_ms:.1f} ms")
print(f"  95th percentile:    {p95_ms:.1f} ms")
print(f"  99th percentile:    {p99_ms:.1f} ms")
print("\nAchievable FPS:")
print(f"  Max theoretical:    {avg_fps:.1f} FPS")
print(f"  With 2x frame skip: {avg_fps * 2:.1f} FPS (process every 2nd frame)")
print(f"  With 3x frame skip: {avg_fps * 3:.1f} FPS (process every 3rd frame)")
print("="*50)

# Compare with video FPS
if avg_fps < fps_video:
    print(f"\n⚠️ Model is slower than video FPS ({fps_video:.1f}).")
    print("   Use frame skipping to maintain real-time display.")
else:
    print(f"\n✅ Model is faster than video FPS ({fps_video:.1f}).")
    print("   Real-time processing is possible.")