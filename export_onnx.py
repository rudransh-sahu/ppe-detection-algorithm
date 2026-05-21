"""
Export YOLO model to ONNX with fixed image size.
Run this script from the project root.
"""

from ultralytics import YOLO
import os
import sys

def main():
    model_path = "models/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Make sure you are running from the correct directory.")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Export to ONNX with optimizations for CPU
    print("Exporting to ONNX (image size 416)...")
    model.export(format="onnx", imgsz=416, half=False, opset=12)
    
    # Check output
    onnx_path = "models/weights/best.onnx"
    if os.path.exists(onnx_path):
        print(f"SUCCESS: ONNX model saved at {onnx_path}")
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"File size: {file_size:.2f} MB")
    else:
        print("ERROR: Export failed, ONNX file not found.")

if __name__ == "__main__":
    main()