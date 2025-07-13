import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import mediapipe as mp

def initialize_pose_detector():

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,          
        model_complexity=2,              
        enable_segmentation=False,      
        min_detection_confidence=0.5     
    )
    
    return pose, mp_pose

def is_frontal_face(landmarks, mp_pose):
    
   
    if not landmarks or not landmarks.landmark:
        return False, 1.0
    
    
    lm = landmarks.landmark
    
    
    face_points = [
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.LEFT_EYE.value,
        mp_pose.PoseLandmark.RIGHT_EYE.value,
        mp_pose.PoseLandmark.LEFT_EAR.value,
        mp_pose.PoseLandmark.RIGHT_EAR.value,
    ]
    
    
    if (lm[mp_pose.PoseLandmark.NOSE.value].visibility < 0.7 or
        lm[mp_pose.PoseLandmark.LEFT_EYE.value].visibility < 0.7 or
        lm[mp_pose.PoseLandmark.RIGHT_EYE.value].visibility < 0.7):
        return False, 1.0
    
    
    nose = lm[mp_pose.PoseLandmark.NOSE.value]
    left_eye = lm[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = lm[mp_pose.PoseLandmark.RIGHT_EYE.value]
    left_ear = lm[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = lm[mp_pose.PoseLandmark.RIGHT_EAR.value]
    
    
    eye_diff_y = abs(left_eye.y - right_eye.y)
    
    
    eye_width = abs(right_eye.x - left_eye.x)
    
    
    nose_center_diff = abs(nose.x - (left_eye.x + right_eye.x) / 2)
    
    ear_visibility_diff = abs(left_ear.visibility - right_ear.visibility)
    
    face_score = (
        0.3 * eye_diff_y +          
        0.3 * nose_center_diff +     
        0.4 * ear_visibility_diff    
    )
    
    if ((left_ear.visibility > 0.8 and right_ear.visibility < 0.3) or 
        (right_ear.visibility > 0.8 and left_ear.visibility < 0.3)):
        face_score += 0.5
    
    if eye_width < 0.08:
        face_score += 0.3
    
    is_frontal = (
        face_score < 0.3 and        
        eye_diff_y < 0.05 and       
        eye_width > 0.08 and        
        ear_visibility_diff < 0.4   
    )
    
    return is_frontal, face_score

def get_image_quality(image_path):
    
    img = cv2.imread(image_path)
    if img is None:
        return 0
    
    
    height, width = img.shape[:2]
    resolution_score = height * width
    
   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    
    quality_score = resolution_score * (0.5 + 0.5 * min(lap_var/100, 1.0))
    
    return quality_score


def filter_person_images_per5(person_dir, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing the MediaPipe Pose detector...")
    pose_detector, mp_pose = initialize_pose_detector()
    
    image_files = []
    for ext in ["*.jpg", "*.png", "*.jpeg", "*.webp"]:
        image_files.extend(list(Path(person_dir).glob(ext)))
    
    person_groups = {}
    for img_path in image_files:
        parts = img_path.stem.split('_')
        if len(parts) >= 1:
            person_id = parts[0]
            if person_id not in person_groups:
                person_groups[person_id] = []
            person_groups[person_id].append(str(img_path))
    
    print(f"A total of {len(person_groups)} different pedestrian IDs were found")
    print(f"A total of {len(image_files)} images were found")
    
    
    count_saved = 0
    
    for person_id, images in tqdm(person_groups.items(), desc="处理行人"):
        
        best_images = []
        image_resolutions = []
        
        for img_path in images:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
            
                height, width = img.shape[:2]
                resolution = height * width
                # if is_front:
                image_resolutions.append((img_path, resolution))
            except Exception as e:
                continue
   
        image_resolutions.sort(key=lambda x: x[1], reverse=True)

      
        top_images = image_resolutions[:min(10, len(image_resolutions))]
      
        if len(top_images) >= 4:
            import random
           
            selected_images = random.sample(top_images, 4)
            
           
            for img_path, _ in selected_images:
                
                quality = get_image_quality(img_path)
                best_images.append((img_path, quality))
        else:
         
            for img_path, _ in top_images:
                quality = get_image_quality(img_path)
                best_images.append((img_path, quality))

        

        if best_images:
            
           
            for img in best_images:
                parent_dir = os.path.basename(os.path.dirname(img[0]))
                basename = os.path.basename(os.path.basename(img[0]))
                
            
      
                shutil.copy(img[0], os.path.join(output_dir, basename))
                count_saved += 1
      
            print(f"Best image saved for ID {person_id}: {os.path.join(output_dir, basename)}")


        else:
            print(f"The image with ID {person_id} was not found or the processing failed.")
    
    print(f"The best frontal face images of {count_saved}/{len(person_groups)} pedestrians have been filtered and saved.")
    print(f"The results have been saved to {output_dir}")


if __name__ == "__main__":
  
    person_dir = "/path/to/LTCC_ReID/train"  
    output_dir = "/path/to/output_dir"
    
   
    filter_person_images_per5(person_dir, output_dir)