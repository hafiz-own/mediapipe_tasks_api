import numpy as np
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from mediapipe.tasks.python import vision
from utils.draw_core import get_scale_factors, scale_connection_spec

def draw_face_landmarks(rgb_image, face_landmarks_list):
    annotated_image = np.copy(rgb_image)
    if not face_landmarks_list:
        return annotated_image

    line_scale, dot_scale = get_scale_factors(rgb_image.shape)
    
    # Face mesh is extremely dense, so we drastically reduce its line thickness
    face_line_scale = max(1, line_scale // 3)
    
    # Handle both Holistic (single list) and FaceLandmarker (list of lists)
    if hasattr(face_landmarks_list[0], 'x'):
        face_landmarks_list = [face_landmarks_list]

    for face_landmarks in face_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=scale_connection_spec(drawing_styles.get_default_face_mesh_tesselation_style(), face_line_scale)
        )
    return annotated_image

def draw_face_result(rgb_image, detection_result):
    if not detection_result.face_landmarks:
        return rgb_image
    return draw_face_landmarks(rgb_image, detection_result.face_landmarks)
