import numpy as np
import cv2
from utils.draw_face import draw_face_landmarks
from utils.draw_pose import draw_pose_landmarks
from utils.draw_hands import draw_hand_landmarks

# RGB colours for the hand connection lines, so left and right are easy to tell apart.
# "Left"/"right" are the signer's own hands (MediaPipe's convention), not image sides.
LEFT_HAND_COLOR = (255, 140, 0)   # orange
RIGHT_HAND_COLOR = (0, 190, 255)  # light blue


def _draw_legend(image):
    """Small 'left hand / right hand' key in the top-left corner."""
    h, w = image.shape[:2]
    scale = max(0.5, w / 1600)
    thick = max(1, int(round(scale * 2)))
    y = int(30 * scale)
    for text, color in (("left hand", LEFT_HAND_COLOR), ("right hand", RIGHT_HAND_COLOR)):
        cv2.putText(image, text, (int(10 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(image, text, (int(10 * scale), y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        y += int(32 * scale)
    return image


def draw_holistic_landmarks(rgb_image, detection_result):
    """Draws holistic landmarks on an RGB image by delegating to modular functions."""
    annotated_image = np.copy(rgb_image)

    if detection_result.face_landmarks:
        annotated_image = draw_face_landmarks(annotated_image, detection_result.face_landmarks)

    if detection_result.pose_landmarks:
        annotated_image = draw_pose_landmarks(annotated_image, detection_result.pose_landmarks)

    if detection_result.left_hand_landmarks:
        annotated_image = draw_hand_landmarks(annotated_image, detection_result.left_hand_landmarks, LEFT_HAND_COLOR)
    if detection_result.right_hand_landmarks:
        annotated_image = draw_hand_landmarks(annotated_image, detection_result.right_hand_landmarks, RIGHT_HAND_COLOR)

    return _draw_legend(annotated_image)
