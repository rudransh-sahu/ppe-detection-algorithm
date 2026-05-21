#!/usr/bin/env python3
"""
train_yolo.py - PPE Detection Model Training Script

Description:
    Trains a YOLO11n (or YOLOv8n) model on a custom PPE dataset for edge deployment.
    Includes advanced augmentations specifically for heavy industrial environments
    with dust, backlighting, and varying camera angles.

Usage:
    python train_yolo.py                    # Standard training
    python train_yolo.py --resume           # Resume interrupted training
    python train_yolo.py --epochs 100       # Override default epochs
    python train_yolo.py --model yolov8n    # Use YOLOv8n instead of YOLO11n
"""

import argparse
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
# Paths to your dataset and model weights
DATA_CONFIG = "data/raw/data.yaml"    # YAML file describing dataset
MODEL_NAME = "yolo11n.pt"             # Base model (lightweight)
OUTPUT_DIR = "runs/train/ppe"         # Directory for training outputs
EPOCHS = 50                           # Number of training epochs
IMAGE_SIZE = 640                      # Input image size (pixels)
BATCH_SIZE = 16                       # Batch size (adjust based on GPU memory)
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
# ============================================================
# DATA AUGMENTATION FOR INDUSTRIAL ENVIRONMENTS
# ============================================================
# These augmentations simulate heavy industrial conditions (dust, backlighting,
# low light, distorted angles, and reflections).
# Adjust these parameters based on your site's visual characteristics.

AUGMENTATIONS = {
    # ----- HSV Color Space Augmentations (Handle Dust & Backlighting) -----
    # Hue: Color shift (dust, smoke, abnormal lighting)
    # Saturation: Vibrance changes (washed-out, over-saturated)
    # Value: Brightness variation (backlighting, shadows)
    'hsv_h': 0.05,     # Hue variation (default: 0.015)
    'hsv_s': 0.8,      # Saturation variation (default: 0.7)
    'hsv_v': 0.5,      # Value/Brightness variation (default: 0.4)

    # ----- Geometric Augmentations (Handle Varying Camera Angles) -----
    # Rotation: Camera tilt or rolled angles
    # Scale: Zoom variations (objects larger/smaller)
    # Shear: Skewed perspectives
    # Perspective: Extreme viewpoint changes (side-angle cams)
    'degrees': 15.0,   # Rotation (+/- degrees)
    'translate': 0.2,   # Translation (shift as fraction of image size)
    'scale': 0.5,       # Scaling (+/- factor)
    'shear': 10.0,      # Shear degrees
    'perspective': 0.001, # Perspective distortion (0 to 0.001)

    # ----- Flipping Augmentations (Camera Mirroring) -----
    'flipud': 0.0,      # Upside-down flip (rare in fixed cams)
    'fliplr': 0.5,      # Left-right flip (common for side-mounted cams)

    # ----- Advanced Augmentations (Improve Generalization) -----
    'mosaic': 1.0,      # Mosaic augmentation (combine 4 images)
    'mixup': 0.2,       # Mixup augmentation (blend 2 images)
    'copy_paste': 0.1,  # Copy-paste augmentation (for overlapping objects)
}

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(resume=False, epochs=EPOCHS, model_name=MODEL_NAME):
    """
    Train a YOLO model with custom augmentations.

    Args:
        resume (bool): Resume training from last checkpoint
        epochs (int): Number of training epochs
        model_name (str): Base model filename (e.g., 'yolo11n.pt', 'yolov8n.pt')
    """
    # Load pre-trained model
    print(f"🚀 Loading base model: {model_name}")
    model = YOLO(model_name)

    # Training arguments
    train_args = {
        'data': DATA_CONFIG,          # Path to dataset YAML
        'epochs': epochs,             # Number of epochs
        'imgsz': IMAGE_SIZE,          # Input image size
        'batch': BATCH_SIZE,          # Batch size
        'device': DEVICE,             # GPU/CPU device
        'workers': 8,                 # Number of dataloader workers
        'project': OUTPUT_DIR,        # Output directory
        'name': 'exp',                # Experiment name
        'exist_ok': True,             # Overwrite existing experiment
        'pretrained': True,           # Start from pretrained weights
        'optimizer': 'AdamW',         # Optimizer (AdamW for stability)
        'lr0': 0.01,                  # Initial learning rate
        'lrf': 0.01,                  # Final learning rate factor
        'momentum': 0.937,            # SGD momentum (if using SGD)
        'weight_decay': 0.0005,       # Weight decay for regularization
        'warmup_epochs': 3.0,         # Warmup epochs
        'warmup_momentum': 0.8,       # Warmup initial momentum
        'warmup_bias_lr': 0.1,        # Warmup bias learning rate
        'box': 7.5,                   # Box loss gain
        'cls': 0.5,                   # Class loss gain
        'dfl': 1.5,                   # DFL loss gain
        'pose': 12.0,                 # Pose loss gain (for pose models)
        'kobj': 1.0,                  # Keypoint objectness loss gain
        'label_smoothing': 0.0,       # Label smoothing factor
        'nbs': 64,                    # Nominal batch size
        'overlap_mask': True,         # Masks should overlap
        'mask_ratio': 4,              # Mask downsample ratio
        'dropout': 0.0,               # Dropout rate
        'val': True,                  # Validate during training
        'save': True,                 # Save checkpoints
        'save_period': -1,            # Save checkpoint every N epochs (-1 = disabled)
        'cache': True,                # Cache images for faster training
        'seed': 42,                   # Random seed for reproducibility
        'verbose': True,              # Print detailed logs
        'patience': 100,              # Early stopping epochs
        'resume': resume,             # Resume training
    }

    # Merge augmentations into training arguments
    train_args.update(AUGMENTATIONS)

    # Start training
    print(f"📊 Starting training for {epochs} epochs...")
    print(f"🎯 Dataset: {DATA_CONFIG}")
    print(f"🔧 Augmentations: {list(AUGMENTATIONS.keys())}")

    model.train(**train_args)

    # After training, export the model for inference
    print("✅ Training complete! Exporting model...")
    model_path = Path(OUTPUT_DIR) / "exp" / "weights" / "best.pt"
    if model_path.exists():
        # Export to ONNX format for faster inference
        model.export(format="onnx", imgsz=IMAGE_SIZE)
        print(f"🎉 Model exported to ONNX format.")

    print(f"🏁 Training finished. Best model saved at: {model_path}")
    return model_path

# ============================================================
# VALIDATION FUNCTION
# ============================================================
def validate_model(model_path):
    """
    Validate a trained YOLO model on the validation set.

    Args:
        model_path (str or Path): Path to the trained model weights
    """
    print(f"🔍 Validating model: {model_path}")
    model = YOLO(model_path)
    results = model.val(data=DATA_CONFIG, device=DEVICE)

    # Print key metrics
    print("\n📈 Validation Results:")
    print(f"   - mAP50-95: {results.box.map:.4f}")
    print(f"   - mAP50:    {results.box.map50:.4f}")
    print(f"   - Precision: {results.box.mp:.4f}")
    print(f"   - Recall:    {results.box.mr:.4f}")
    return results

# ============================================================
# MAIN ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO for PPE Detection")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        choices=["yolo11n.pt", "yolov8n.pt"],
                        help="Base model to train (yolo11n.pt or yolov8n.pt)")
    parser.add_argument("--validate", action="store_true",
                        help="Validate an existing model")
    parser.add_argument("--weights", type=str,
                        help="Path to trained weights for validation (e.g., runs/train/ppe/exp/weights/best.pt)")
    args = parser.parse_args()

    if args.validate and args.weights:
        validate_model(args.weights)
    else:
        trained_model_path = train_model(
            resume=args.resume,
            epochs=args.epochs,
            model_name=args.model
        )
        # Optionally validate after training
        if trained_model_path:
            validate_model(trained_model_path)