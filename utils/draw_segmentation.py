import numpy as np
import cv2


def draw_segmentation_result(rgb_image, detection_result, visual_mode="blur"):
    """
    Draws the segmentation result onto the RGB image.
    Supports 'blur', 'desaturate', and 'overlay' visual modes.
    """
    if detection_result.category_mask is None:
        return rgb_image

    # 1. Person mask (True = person/foreground).
    # The selfie_segmenter returns category 0 for the person and >0 for the background.
    category = detection_result.category_mask.numpy_view().squeeze()  # strictly (H, W)
    mask = category == 0
    condition = np.stack((mask,) * 3, axis=-1)

    if visual_mode == "blur":
        # Keep the person sharp, blur the background heavily
        blurred_bg = cv2.GaussianBlur(rgb_image, (55, 55), 0)
        return np.where(condition, rgb_image, blurred_bg)

    elif visual_mode == "desaturate":
        # Keep the person in colour, grayscale the background
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        gray_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return np.where(condition, rgb_image, gray_3c)

    elif visual_mode == "overlay":
        # Colour heatmap of the category values, blended over the PERSON (category 0)
        # region only; the background is left as the original image.
        # Multiply by 50 to spread the colormap out when there are few classes.
        mask_uint8 = (category * 50).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(rgb_image, 0.5, heatmap, 0.5, 0)
        return np.where(condition, overlay, rgb_image)

    return rgb_image
