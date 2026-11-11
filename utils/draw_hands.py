import numpy as np
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from mediapipe.tasks.python import vision
from utils.draw_core import get_scale_factors, scale_landmark_spec, scale_connection_spec


def draw_hand_landmarks(rgb_image, hand_landmarks_list, line_color=None):
    """Draw one or more hands.

    line_color: optional RGB tuple. If given, all hand connections are drawn in that
    single colour (used by holistic to tell left and right apart); otherwise MediaPipe's
    default per-finger colours are used.
    """
    annotated_image = np.copy(rgb_image)
    if not hand_landmarks_list:
        return annotated_image

    line_scale, dot_scale = get_scale_factors(rgb_image.shape)

    # Nudge down the hand line thickness slightly so it's not as thick as the pose
    hand_line_scale = max(1, int(line_scale * 0.7))

    if line_color is None:
        connection_spec = drawing_styles.get_default_hand_connections_style()
    else:
        connection_spec = drawing_utils.DrawingSpec(color=line_color, thickness=2)

    # Handle both Holistic (single list) and HandLandmarker (list of lists)
    if hasattr(hand_landmarks_list[0], 'x'):
        hand_landmarks_list = [hand_landmarks_list]

    for hand_landmarks in hand_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=hand_landmarks,
            connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=scale_landmark_spec(drawing_styles.get_default_hand_landmarks_style(), dot_scale),
            connection_drawing_spec=scale_connection_spec(connection_spec, hand_line_scale)
        )
    return annotated_image


def draw_hand_result(rgb_image, detection_result):
    if not detection_result.hand_landmarks:
        return rgb_image
    return draw_hand_landmarks(rgb_image, detection_result.hand_landmarks)
