# test_model.py
from ultralytics import YOLO
import cv2

# Load model
model = YOLO("models/weights/best.pt")  # adjust path

# Run on an image
results = model(r"D:\opencv\construction_worker.jpg", conf=0.5)

# Show results
annotated = results[0].plot()
cv2.imshow("Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()