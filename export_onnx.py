"""
Export YOLO model to ONNX for use in the PPE detection pipeline.
Optimized for imgsz=320 (best speed/accuracy trade-off on CPU).
"""

from ultralytics import YOLO
import os
import sys

def main():
    # Paths
    model_pt = "models/weights/best.pt"
    model_onnx = "models/weights/best.onnx"
    
    # Check if source model exists
    if not os.path.exists(model_pt):
        print(f"ERROR: PyTorch model not found at {model_pt}")
        print("Please train or download best.pt first.")
        sys.exit(1)
    
    # Optional: remove old ONNX if it exists (to avoid confusion)
    if os.path.exists(model_onnx):
        print(f"Removing existing {model_onnx} ...")
        os.remove(model_onnx)
    
    print(f"Loading model from {model_pt} ...")
    model = YOLO(model_pt)
    
    # Export to ONNX with settings matching your pipeline
    print("Exporting to ONNX (image size 320) ...")
    try:
        model.export(
            format="onnx",
            imgsz=320,          # matches final pipeline (fast)
            half=False,         # keep full precision for reliability
            opset=12,           # widely supported
            dynamic=False,      # fixed input size (faster)
            simplify=True       # optional, reduces model size
        )
    except Exception as e:
        print(f"Export failed: {e}")
        sys.exit(1)
    
    # Verify output
    if os.path.exists(model_onnx):
        file_size = os.path.getsize(model_onnx) / (1024 * 1024)
        print(f"SUCCESS: ONNX model saved at {model_onnx}")
        print(f"File size: {file_size:.2f} MB")
        print("\nYou can now run the pipeline using this ONNX model.")
    else:
        print("ERROR: Export failed, ONNX file not created.")
        sys.exit(1)

if __name__ == "__main__":
    main()