import os
import cv2
import json
import argparse
import joblib
import numpy as np
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "best_pose_classifier.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "saved_models", "label_encoder.pkl")
IDEAL_POSES_JSON = os.path.join(BASE_DIR, "dataset", "ideal_poses.json")
TASK_MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_full.task")

# Standard MediaPipe 33 Pose Connections
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

# Clean display names mapping
POSE_NAME_MAP = {
    "Akarna -Dhnurasan": "Akarna Dhanurasan (Archer Pose)",
    "Bagali_Phara": "Bagali Phara (Armpit Latch)",
    "Bajarang_Pakad": "Bajarang Pakad (Hanuman Clasp)",
    "Bandar_Pakad": "Bandar Pakad (Monkey Hold)",
    "Baneguruji_Pakad": "Baneguruji Pakad (Master Grip)",
    "Chakorasan": "Chakorasan (Wheel Pose)",
    "Guru_Pakad": "Guru Pakad (Traditional Lock)",
    "Hatacaha_Para_Stand": "Hatacha Para Stand (Hand Pole Balance)",
    "Hatacha_Phara": "Hatacha Phara (Forearm Mount)",
    "Kukkutsan": "Kukkutsan (Rooster Balance)",
    "Mayurasan_2_Hand": "Mayurasan (Peacock Balance)",
    "Natarajasan1": "Natarajasan (Dancer Pose)",
    "Natarajasan_Outer_Hand": "Natarajasan Outer Hold",
    "Naukasan": "Naukasan (Boat Pose Mount)",
    "Padmasan_from_hatach_fara": "Padmasan (Lotus Pole Mount)",
    "Padmasan_one_hand _hold": "Padmasan One-Hand Hold",
    "Patanagi": "Patanagi (Kite Flag Pose)",
    "Paticha_Tajawa": "Paticha Tajawa (Back Balance)",
    "StraddleL_Hold": "Straddle L-Hold (Core Balance)"
}

_detector = None

def get_pose_detector():
    global _detector
    if _detector is None:
        base_options = python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.25,
            min_pose_presence_confidence=0.25
        )
        _detector = vision.PoseLandmarker.create_from_options(options)
    return _detector

POLE_MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "yolov8_pole.pt")
yolo_pole_model = None

try:
    from ultralytics import YOLO
    os.environ["YOLO_VERBOSE"] = "False"
    yolo_model = YOLO('yolov8n.pt')
    if os.path.exists(POLE_MODEL_PATH):
        yolo_pole_model = YOLO(POLE_MODEL_PATH)
        print(f"Loaded custom YOLOv8 Pole Detection Model from: {POLE_MODEL_PATH}")
except Exception as e:
    yolo_model = None
    yolo_pole_model = None

def auto_zoom_person(image_bgr):
    if yolo_model is None:
        return image_bgr
        
    results = yolo_model(image_bgr, classes=[0], verbose=False)
    if not results or len(results[0].boxes) == 0:
        return image_bgr
        
    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    largest_box_idx = np.argmax(areas)
    
    x1, y1, x2, y2 = map(int, boxes[largest_box_idx])
    
    h, w = image_bgr.shape[:2]
    margin_x = int((x2 - x1) * 0.25)
    margin_y = int((y2 - y1) * 0.25)
    
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)
    
    cropped_bgr = image_bgr[y1:y2, x1:x2]
    if cropped_bgr.shape[0] < 50 or cropped_bgr.shape[1] < 50:
        return image_bgr
        
    return cropped_bgr

def normalize_landmarks(pose_landmarks):
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_HIP, RIGHT_HIP = 23, 24

    points = []
    for lm in pose_landmarks:
        vis = getattr(lm, 'visibility', 0.99)
        if vis is None: vis = 0.99
        points.append([lm.x, lm.y, lm.z, vis])
    points = np.array(points)

    hip_center_x = (points[LEFT_HIP][0] + points[RIGHT_HIP][0]) / 2.0
    hip_center_y = (points[LEFT_HIP][1] + points[RIGHT_HIP][1]) / 2.0
    hip_center_z = (points[LEFT_HIP][2] + points[RIGHT_HIP][2]) / 2.0
    points[:, 0] -= hip_center_x
    points[:, 1] -= hip_center_y
    points[:, 2] -= hip_center_z

    shoulder_center_x = (points[LEFT_SHOULDER][0] + points[RIGHT_SHOULDER][0]) / 2.0
    shoulder_center_y = (points[LEFT_SHOULDER][1] + points[RIGHT_SHOULDER][1]) / 2.0
    shoulder_center_z = (points[LEFT_SHOULDER][2] + points[RIGHT_SHOULDER][2]) / 2.0

    torso_size = math.sqrt(
        (shoulder_center_x - 0)**2 + 
        (shoulder_center_y - 0)**2 + 
        (shoulder_center_z - 0)**2
    )

    if torso_size > 0:
        points[:, 0] /= torso_size
        points[:, 1] /= torso_size
        points[:, 2] /= torso_size

    return points.flatten().tolist()

def detect_pole(image_bgr, pose_landmarks=None):
    """
    Dual-Engine Pole Detection Pipeline:
    Engine 1 (Primary): Custom-trained YOLOv8 Pole Object Detector (`yolov8_pole.pt`).
    Engine 2 (Fallback): Geometric line analysis + MediaPipe limb masking + HSV wood/rope color validation.
    """
    if image_bgr is None or image_bgr.size == 0:
        return False

    # 1. Primary Deep Learning Check (YOLOv8 Pole Model)
    if yolo_pole_model is not None:
        try:
            results = yolo_pole_model(image_bgr, conf=0.25, verbose=False)
            if results and len(results[0].boxes) > 0:
                return True
        except Exception as e:
            pass

    # 2. Fallback Heuristic Check
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)

    # 1. Mask out human body edges using MediaPipe landmarks if available
    if pose_landmarks:
        body_mask = np.ones((h, w), dtype=np.uint8) * 255
        limb_thickness = int(max(18, min(w, h) * 0.05))  # Mask thickness proportional to image
        
        coords = []
        for lm in pose_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            coords.append((cx, cy))
            
        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(coords) and end_idx < len(coords):
                p1, p2 = coords[start_idx], coords[end_idx]
                cv2.line(body_mask, p1, p2, 0, limb_thickness)
                
        # Also mask out head/face bounding region
        if len(coords) > 0:
            nose = coords[0]
            head_radius = int(max(25, min(w, h) * 0.08))
            cv2.circle(body_mask, nose, head_radius, 0, -1)

        # Apply body mask to edge map
        edges = cv2.bitwise_and(edges, edges, mask=body_mask)

    # 2. Lower min line length to handle pole occlusion by athlete body
    min_line_len = int(h * 0.12)  # Reduced from 0.25 to 0.12 to detect pole segments broken by athlete body
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=45, minLineLength=min_line_len, maxLineGap=40)

    if lines is None:
        return False

    # Extract vertical line candidates
    vertical_segments = []
    for line in lines:
        x1, y1, x2, y2 = line.flatten()[:4]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy == 0:
            continue
        
        slope_ratio = dx / float(dy)
        line_len = math.sqrt(dx*dx + dy*dy)

        # Steep angle check (slope ratio < 0.32 means within ~18 degrees of vertical)
        if slope_ratio < 0.32 and line_len >= min_line_len:
            avg_x = (x1 + x2) / 2.0
            
            # If pose landmarks present, require pole to be within athlete reach (within 45% of image width)
            if pose_landmarks:
                xs = [lm.x * w for lm in pose_landmarks]
                person_center_x = sum(xs) / len(xs)
                if abs(avg_x - person_center_x) > (w * 0.45):
                    continue
                    
            vertical_segments.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'avg_x': avg_x, 'len': line_len,
                'min_y': min(y1, y2), 'max_y': max(y1, y2)
            })

    if not vertical_segments:
        return False

    # Group vertical segments by X axis alignment (clustering lines along the same vertical column)
    axis_clusters = []
    x_tolerance = w * 0.04  # Lines within 4% of image width belong to the same pole axis
    
    for seg in vertical_segments:
        matched = False
        for cluster in axis_clusters:
            cluster_avg_x = sum(s['avg_x'] for s in cluster) / len(cluster)
            if abs(seg['avg_x'] - cluster_avg_x) < x_tolerance:
                cluster.append(seg)
                matched = True
                break
        if not matched:
            axis_clusters.append([seg])

    # Evaluate clusters: A valid pole has total vertical coverage or parallel edge pairs
    hsv_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    for cluster in axis_clusters:
        # Calculate combined vertical span of this cluster
        min_y = min(s['min_y'] for s in cluster)
        max_y = max(s['max_y'] for s in cluster)
        total_span = max_y - min_y
        total_line_len = sum(s['len'] for s in cluster)
        
        # Check if total line coverage is at least 20% of image height
        if total_span >= (h * 0.20) or total_line_len >= (h * 0.20):
            # HSV Wood / Tan Color Check around the detected axis
            cluster_x = int(sum(s['avg_x'] for s in cluster) / len(cluster))
            crop_x1 = max(0, cluster_x - int(w * 0.03))
            crop_x2 = min(w, cluster_x + int(w * 0.03))
            crop_y1 = max(0, min_y)
            crop_y2 = min(h, max_y)
            
            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                hsv_roi = hsv_img[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Wood/tan/brown/rope HSV ranges:
                # Brown/Wood: Hue 4-35, Sat 15-220, Val 30-240
                # Off-white/Rope: Sat < 50, Val > 100
                h_ch, s_ch, v_ch = hsv_roi[:, :, 0], hsv_roi[:, :, 1], hsv_roi[:, :, 2]
                
                wood_mask = ((h_ch >= 4) & (h_ch <= 35) & (s_ch >= 15) & (v_ch >= 30)) | \
                            ((s_ch <= 50) & (v_ch >= 100))  # Rope / pale wood
                
                match_ratio = np.mean(wood_mask)
                
                # If color matches wooden pole/rope characteristics (> 30% of ROI pixels)
                if match_ratio >= 0.30:
                    return True
            else:
                return True

    # Check if any two distinct clusters form parallel edges of a pole (separated by pole width)
    if len(axis_clusters) >= 2:
        cluster_xs = [sum(s['avg_x'] for s in c) / len(c) for c in axis_clusters]
        for i in range(len(cluster_xs)):
            for j in range(i + 1, len(cluster_xs)):
                dist = abs(cluster_xs[i] - cluster_xs[j])
                # Pole width is typically 1.5% to 12% of image width
                if (w * 0.015) <= dist <= (w * 0.12):
                    return True

    return False

def calculate_accuracy(user_angles, ideal_angles):
    """ Honest accuracy scoring formula based on joint angle error """
    errors = [abs(user_angles[k] - ideal_angles[k]) for k in user_angles.keys()]
    avg_error = sum(errors) / len(errors)
    score = max(0.0, 100.0 - (avg_error * 2.0))
    return min(98.5, score)

def get_grade(score):
    if score >= 85: return "Excellent"
    elif score >= 72: return "Very Good"
    elif score >= 60: return "Good"
    else: return "Needs Improvement"

def calculate_3d_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_angles(features):
    pts = np.array(features).reshape(33, 4)[:, :3]
    return {
        "Left Elbow": calculate_3d_angle(pts[11], pts[13], pts[15]),
        "Right Elbow": calculate_3d_angle(pts[12], pts[14], pts[16]),
        "Left Shoulder": calculate_3d_angle(pts[23], pts[11], pts[13]),
        "Right Shoulder": calculate_3d_angle(pts[24], pts[12], pts[14]),
        "Left Hip": calculate_3d_angle(pts[11], pts[23], pts[25]),
        "Right Hip": calculate_3d_angle(pts[12], pts[24], pts[26]),
        "Left Knee": calculate_3d_angle(pts[23], pts[25], pts[27]),
        "Right Knee": calculate_3d_angle(pts[24], pts[26], pts[28])
    }

def generate_feedback(user_angles, ideal_angles, threshold=25):
    feedback = []
    for joint, u_angle in user_angles.items():
        diff = u_angle - ideal_angles[joint]
        if abs(diff) > threshold:
            if diff > 0:
                feedback.append(f"Bend your {joint} slightly more.")
            else:
                feedback.append(f"Straighten your {joint} for optimal form.")
    return feedback if feedback else ["Perfect pose alignment! Excellent posture."]

def compute_cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

def draw_skeleton(image_rgb, pose_landmarks):
    h, w, _ = image_rgb.shape
    coords = []
    for lm in pose_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(coords) and end_idx < len(coords):
            p1, p2 = coords[start_idx], coords[end_idx]
            cv2.line(image_rgb, p1, p2, (255, 255, 255), 3)

    for cx, cy in coords:
        cv2.circle(image_rgb, (cx, cy), 5, (255, 94, 0), -1)
        cv2.circle(image_rgb, (cx, cy), 6, (255, 255, 255), 1)

def process_image_array(image_bgr, model, encoder, ideal_poses, fast_mode=False):
    detector = get_pose_detector()
    
    # Process image directly in original orientation
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    detection_result = detector.detect(mp_image)
    
    if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
        return {"error": "No human body detected in image. Ensure athlete's full body is visible.", "image": image_rgb}
        
    pose_landmarks = detection_result.pose_landmarks[0]
    features = normalize_landmarks(pose_landmarks)
    probabilities = model.predict_proba([features])[0]
    max_prob_index = np.argmax(probabilities)
    raw_confidence = float(probabilities[max_prob_index])
    
    # Draw skeleton overlay
    draw_skeleton(image_rgb, pose_landmarks)
    pole_detected = detect_pole(image_bgr, pose_landmarks)
    
    raw_class_name = encoder.classes_[max_prob_index]
    display_pose_name = POSE_NAME_MAP.get(raw_class_name, raw_class_name.replace("_", " "))
    user_angles = extract_angles(features)
    
    # Rejection threshold for non-human / completely garbage poses
    CONFIDENCE_THRESHOLD = 0.25
    
    if raw_confidence < CONFIDENCE_THRESHOLD:
        return {
            "unrecognized": True,
            "pose": "Unrecognized / Non-Mallakhamb Pose",
            "raw_pose": "Unrecognized",
            "confidence": round(raw_confidence * 100.0, 1),
            "accuracy": 0.0,
            "grade": "Needs Improvement",
            "feedback": [
                "The AI engine did not detect a recognized Mallakhamb posture on a pole/rope.",
                "Ensure the athlete is performing a clear Mallakhamb pose on a vertical pole or rope."
            ],
            "poleDetected": pole_detected,
            "image": image_rgb
        }

    ideal_features = ideal_poses.get(raw_class_name)
    
    if ideal_features:
        if isinstance(ideal_features[0], list):
            best_accuracy = -1
            best_ideal_angles = None
            for ideal_feat in ideal_features:
                cos_sim = compute_cosine_similarity(features, ideal_feat)
                ideal_angles = extract_angles(ideal_feat)
                angle_acc = calculate_accuracy(user_angles, ideal_angles)
                combined = (cos_sim * 60.0) + (angle_acc * 0.40)
                if combined > best_accuracy:
                    best_accuracy = combined
                    best_ideal_angles = ideal_angles
            ideal_angles = best_ideal_angles
            accuracy_score = round(min(98.5, max(45.0, best_accuracy)), 1)
        else:
            cos_sim = compute_cosine_similarity(features, ideal_features)
            ideal_angles = extract_angles(ideal_features)
            angle_acc = calculate_accuracy(user_angles, ideal_angles)
            combined = (cos_sim * 60.0) + (angle_acc * 0.40)
            accuracy_score = round(min(98.5, max(45.0, combined)), 1)
            
        grade = get_grade(accuracy_score)
        feedback_list = generate_feedback(user_angles, ideal_angles, threshold=25)
    else:
        accuracy_score = round(raw_confidence * 100.0, 1)
        grade = get_grade(accuracy_score)
        feedback_list = ["Maintain steady posture on pole."]

    honest_confidence = round(raw_confidence * 100.0, 1)

    return {
        "pose": display_pose_name,
        "raw_pose": raw_class_name,
        "confidence": honest_confidence,
        "accuracy": accuracy_score,
        "grade": grade,
        "feedback": feedback_list,
        "poleDetected": pole_detected,
        "image": image_rgb
    }


