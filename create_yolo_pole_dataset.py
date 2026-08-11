import os
import cv2
import glob
import math
import random
import shutil
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROW_DATASET_DIR = os.path.join(BASE_DIR, "row_dataset")
OUTPUT_YOLO_DIR = os.path.join(BASE_DIR, "yolo_pole_dataset")
TASK_MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_full.task")

# Initialize MediaPipe Pose Landmarker
base_options = python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.25
)
detector = vision.PoseLandmarker.create_from_options(options)

POSE_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15),
    (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28)
]

def auto_detect_pole_bbox(image_bgr):
    """
    Automatically calculates a tight YOLO bounding box [x_center, y_center, width, height]
    for the Mallakhamb pole in an image using geometric line clustering and body landmark anchoring.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 130)

    # Detect human landmarks using MediaPipe
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(mp_image)

    pose_landmarks = None
    if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
        pose_landmarks = detection_result.pose_landmarks[0]

    # Mask human body limbs if landmarks present
    if pose_landmarks:
        body_mask = np.ones((h, w), dtype=np.uint8) * 255
        limb_thickness = int(max(20, min(w, h) * 0.06))
        coords = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks]
        
        for p1_idx, p2_idx in POSE_CONNECTIONS:
            if p1_idx < len(coords) and p2_idx < len(coords):
                cv2.line(body_mask, coords[p1_idx], coords[p2_idx], 0, limb_thickness)
        
        if len(coords) > 0:
            cv2.circle(body_mask, coords[0], int(max(30, min(w, h) * 0.1)), 0, -1)

        edges = cv2.bitwise_and(edges, edges, mask=body_mask)

    min_line_len = int(h * 0.10)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=min_line_len, maxLineGap=40)

    pole_x_center = None
    y_min_val, y_max_val = 0, h

    if lines is not None:
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line.flatten()[:4]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy == 0: continue
            slope_ratio = dx / float(dy)
            line_len = math.sqrt(dx*dx + dy*dy)
            
            if slope_ratio < 0.35 and line_len >= min_line_len:
                avg_x = (x1 + x2) / 2.0
                vertical_lines.append((avg_x, min(y1, y2), max(y1, y2), line_len))

        if vertical_lines:
            # Cluster vertical lines by X coordinate
            clusters = []
            x_tol = w * 0.05
            for vl in vertical_lines:
                matched = False
                for c in clusters:
                    c_avg = sum(item[0] for item in c) / len(c)
                    if abs(vl[0] - c_avg) < x_tol:
                        c.append(vl)
                        matched = True
                        break
                if not matched:
                    clusters.append([vl])

            # Select largest cluster
            best_cluster = max(clusters, key=lambda c: sum(item[3] for item in c))
            pole_x_center = sum(item[0] for item in best_cluster) / len(best_cluster)
            y_min_val = max(0, int(min(item[1] for item in best_cluster) - h * 0.05))
            y_max_val = min(h, int(max(item[2] for item in best_cluster) + h * 0.05))

    # Fallback to MediaPipe landmark center if lines not found
    if pole_x_center is None and pose_landmarks:
        xs = [lm.x * w for lm in pose_landmarks]
        pole_x_center = sum(xs) / len(xs)

    if pole_x_center is None:
        pole_x_center = w / 2.0

    # Ensure pole spans most of vertical height
    y_min_val = min(y_min_val, int(h * 0.05))
    y_max_val = max(y_max_val, int(h * 0.95))

    pole_width = max(int(w * 0.06), 30)
    x1_box = max(0, int(pole_x_center - pole_width / 2.0))
    x2_box = min(w, int(pole_x_center + pole_width / 2.0))

    # Convert to normalized YOLO format [x_center, y_center, bbox_w, bbox_h]
    box_w = (x2_box - x1_box) / float(w)
    box_h = (y_max_val - y_min_val) / float(h)
    x_center = (x1_box + x2_box) / (2.0 * w)
    y_center = (y_min_val + y_max_val) / (2.0 * h)

    return [x_center, y_center, box_w, box_h]

def main():
    print("Collecting images from row_dataset...")
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    all_images = []
    
    for root, dirs, files in os.walk(ROW_DATASET_DIR):
        for f in files:
            if f.lower().endswith(valid_exts):
                all_images.append(os.path.join(root, f))

    print(f"Found {len(all_images)} total images in row_dataset.")
    
    # Random sample 160 images for training dataset
    random.seed(42)
    selected_images = random.sample(all_images, min(160, len(all_images)))
    print(f"Selected {len(selected_images)} images for auto-labeling.")

    # Create dataset directories
    train_img_dir = os.path.join(OUTPUT_YOLO_DIR, "train", "images")
    train_lbl_dir = os.path.join(OUTPUT_YOLO_DIR, "train", "labels")
    val_img_dir = os.path.join(OUTPUT_YOLO_DIR, "val", "images")
    val_lbl_dir = os.path.join(OUTPUT_YOLO_DIR, "val", "labels")

    if os.path.exists(OUTPUT_YOLO_DIR):
        shutil.rmtree(OUTPUT_YOLO_DIR)

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # 80/20 train/val split
    num_train = int(len(selected_images) * 0.8)
    
    count_success = 0
    for idx, img_path in enumerate(selected_images):
        try:
            img = cv2.imread(img_path)
            if img is None: continue

            bbox = auto_detect_pole_bbox(img)
            
            is_val = idx >= num_train
            target_img_dir = val_img_dir if is_val else train_img_dir
            target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

            base_name = f"pole_img_{idx+1:03d}"
            dest_img_path = os.path.join(target_img_dir, f"{base_name}.jpg")
            dest_lbl_path = os.path.join(target_lbl_dir, f"{base_name}.txt")

            # Save image
            cv2.imwrite(dest_img_path, img)

            # Save YOLO label (Class 0 = pole)
            with open(dest_lbl_path, "w") as f:
                f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

            count_success += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Create data.yaml file
    yaml_content = f"""train: {train_img_dir.replace('\\', '/')}
val: {val_img_dir.replace('\\', '/')}

nc: 1
names: ['pole']
"""
    yaml_path = os.path.join(OUTPUT_YOLO_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Successfully generated YOLO pole dataset with {count_success} images!")
    print(f"Dataset config saved to: {yaml_path}")

if __name__ == "__main__":
    main()
