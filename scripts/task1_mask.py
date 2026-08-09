import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/selfie_segmenter.tflite")

def main():
    # We will use the first frame of one of the PSL videos
    video_path = "/home/own/SignBase_Dataset_001/20260808/20260808_182215.mp4"
    output_path = os.path.join(PROJECT_ROOT, "output/task1_raw_mask.jpg")
    
    # Read the first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read video")
        return
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Configure Segmenter to output confidence masks
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.ImageSegmenterOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=False,
        output_confidence_masks=True
    )
    
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        result = segmenter.segment(mp_image)
        
        # Check what we got
        confidence_masks = result.confidence_masks
        print(f"Number of confidence masks returned: {len(confidence_masks)}")
        
        # For selfie_segmenter, index 0 is usually background, index 1 is person.
        # Let's inspect the shapes.
        for i, mask in enumerate(confidence_masks):
            mask_data = mask.numpy_view()
            print(f"Mask {i}: shape={mask_data.shape}, dtype={mask_data.dtype}, min={mask_data.min()}, max={mask_data.max()}")
            
        # We assume index 1 is the person (or index 0 if it's inverted).
        # We will save both masks just to be safe.
        for i, mask in enumerate(confidence_masks):
            mask_data = mask.numpy_view()
            # Convert float32 [0.0, 1.0] to uint8 [0, 255]
            mask_uint8 = (mask_data * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(PROJECT_ROOT, f"output/task1_mask_index_{i}.jpg"), mask_uint8)
            print(f"Saved mask index {i} to output folder.")

if __name__ == "__main__":
    main()
