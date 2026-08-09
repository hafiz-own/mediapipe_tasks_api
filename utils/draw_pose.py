import numpy as np
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from mediapipe.tasks.python import vision
from utils.draw_core import get_scale_factors, scale_landmark_spec, scale_connection_spec

def draw_pose_landmarks(rgb_image, pose_landmarks_list):
    annotated_image = np.copy(rgb_image)
    if not pose_landmarks_list:
        return annotated_image

    line_scale, dot_scale = get_scale_factors(rgb_image.shape)
    
    # Handle both Holistic (single list) and PoseLandmarker (list of lists)
    if hasattr(pose_landmarks_list[0], 'x'):
        pose_landmarks_list = [pose_landmarks_list]

    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=scale_landmark_spec(drawing_styles.get_default_pose_landmarks_style(), dot_scale)
        )
    return annotated_image
