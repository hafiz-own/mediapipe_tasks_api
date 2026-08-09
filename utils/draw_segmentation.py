import numpy as np
import cv2

def draw_segmentation_result(rgb_image, detection_result, visual_mode="blur"):
    """
    Draws the segmentation result onto the RGB image.
    Supports 'blur', 'desaturate', and 'overlay' visual modes.
    """
    if detection_result.category_mask is None:
        return rgb_image

    # 1. Get the mask (True for Foreground, False for Background)
    # The selfie_segmenter returns 0 for the person (foreground) and >0 for the background.
    mask = detection_result.category_mask.numpy_view() == 0
    mask = mask.squeeze() # Ensure it is strictly (H, W)
    condition = np.stack((mask,) * 3, axis=-1)
    
    if visual_mode == "blur":
        # Blur the background heavily
        blurred_bg = cv2.GaussianBlur(rgb_image, (55, 55), 0)
        return np.where(condition, rgb_image, blurred_bg)
        
    elif visual_mode == "desaturate":
        # Grayscale the background
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        gray_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return np.where(condition, rgb_image, gray_3c)
        
    elif visual_mode == "overlay":
        # Apply a colored heatmap overlay representing different categories
        # Multiply by 50 to spread the colormap out if there are few classes
        mask_uint8 = (detection_result.category_mask.numpy_view() * 50).astype(np.uint8)
        heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
        
        # We only want to overlay the heatmap on the foreground (where mask > 0)
        # For background, we just keep the original image
        overlay = cv2.addWeighted(rgb_image, 0.5, heatmap, 0.5, 0)
        return np.where(condition, overlay, rgb_image)
        
    return rgb_image
