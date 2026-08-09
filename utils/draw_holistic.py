import numpy as np
from utils.draw_face import draw_face_landmarks
from utils.draw_pose import draw_pose_landmarks
from utils.draw_hands import draw_hand_landmarks

def draw_holistic_landmarks(rgb_image, detection_result):
    """Draws holistic landmarks on an RGB image by delegating to modular functions."""
    annotated_image = np.copy(rgb_image)
    
    if detection_result.face_landmarks:
        annotated_image = draw_face_landmarks(annotated_image, detection_result.face_landmarks)
        
    if detection_result.pose_landmarks:
        annotated_image = draw_pose_landmarks(annotated_image, detection_result.pose_landmarks)
        
    if detection_result.left_hand_landmarks:
        annotated_image = draw_hand_landmarks(annotated_image, detection_result.left_hand_landmarks)
    if detection_result.right_hand_landmarks:
        annotated_image = draw_hand_landmarks(annotated_image, detection_result.right_hand_landmarks)
            
    return annotated_image
