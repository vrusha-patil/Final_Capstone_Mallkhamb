import os
import shutil
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, "yolo_pole_dataset", "data.yaml")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
TARGET_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "yolov8_pole.pt")

def main():
    print("Loading pre-trained YOLOv8 Nano model...")
    model = YOLO("yolov8n.pt")
    
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    print("Starting YOLOv8 training on custom Mallakhamb Pole dataset...")
    # Train model for 15 epochs on CPU
    results = model.train(
        data=DATA_YAML,
        epochs=15,
        imgsz=512,
        batch=8,
        workers=2,
        name="yolov8_pole_run",
        project=os.path.join(BASE_DIR, "runs"),
        exist_ok=True,
        verbose=True
    )

    best_weights = os.path.join(BASE_DIR, "runs", "yolov8_pole_run", "weights", "best.pt")
    if os.path.exists(best_weights):
        shutil.copy(best_weights, TARGET_MODEL_PATH)
        print(f"\nTraining Complete! Best model saved to: {TARGET_MODEL_PATH}")
    else:
        print("\nWarning: Could not find best.pt weights file!")

if __name__ == "__main__":
    main()
