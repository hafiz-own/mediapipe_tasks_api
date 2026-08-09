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

class PSLAugmenter:
    def __init__(self, 
                 ema_alpha=0.7, 
                 blur_kernel_size=(21, 21),
                 min_coverage=0.02,
                 max_coverage=0.80,
                 max_variance=0.10,
                 invert_mask=False): # confidence_masks[0] is category 0 (person)
        self.ema_alpha = ema_alpha
        self.blur_kernel_size = blur_kernel_size
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage
        self.max_variance = max_variance
        self.invert_mask = invert_mask
        
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        self.options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_category_mask=False,
            output_confidence_masks=True
        )
        
    def _prepare_bg_frame(self, bg_src, target_shape):
        """Resizes/Crops background to perfectly match target (H, W, C)."""
        target_h, target_w = target_shape[:2]
        bg_h, bg_w = bg_src.shape[:2]
        
        # Center crop the background to match aspect ratio
        target_aspect = target_w / target_h
        bg_aspect = bg_w / bg_h
        
        if bg_aspect > target_aspect:
            # BG is wider, crop sides
            new_w = int(bg_h * target_aspect)
            start_x = (bg_w - new_w) // 2
            cropped = bg_src[:, start_x:start_x+new_w]
        else:
            # BG is taller, crop top/bottom
            new_h = int(bg_w / target_aspect)
            start_y = (bg_h - new_h) // 2
            cropped = bg_src[start_y:start_y+new_h, :]
            
        return cv2.resize(cropped, (target_w, target_h))

    def process_video(self, input_video_path, bg_path, output_path):
        print(f"Starting process_video with input={input_video_path}, bg={bg_path}")
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Load background
        is_bg_video = bg_path.lower().endswith(('.mp4', '.avi', '.mov'))
        bg_cap = None
        
        if is_bg_video:
            bg_cap = cv2.VideoCapture(bg_path)
        elif bg_path == "solid_green":
            static_bg = np.zeros((height, width, 3), dtype=np.uint8)
            static_bg[:] = (0, 255, 0) # BGR
        elif bg_path == "solid_gray":
            static_bg = np.zeros((height, width, 3), dtype=np.uint8)
            static_bg[:] = (128, 128, 128) # BGR
        else:
            static_bg = cv2.imread(bg_path)
            static_bg = self._prepare_bg_frame(static_bg, (height, width, 3))
            
        prev_smoothed_mask = None
        prev_coverage = None
        
        anomalies = []
        frame_idx = 0
        
        with vision.ImageSegmenter.create_from_options(self.options) as segmenter:
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    # 1. Background Frame Fetching
                    if is_bg_video:
                        bg_ret, bg_frame = bg_cap.read()
                        if not bg_ret: # Loop background video if it's shorter
                            bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            _, bg_frame = bg_cap.read()
                        current_bg = self._prepare_bg_frame(bg_frame, frame.shape)
                    else:
                        current_bg = static_bg
                        
                    # 2. Segmentation
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    timestamp_ms = int(frame_idx * (1000 / fps))
                    
                    result = segmenter.segment_for_video(mp_image, timestamp_ms)
                    
                    raw_mask = result.confidence_masks[0].numpy_view().squeeze().copy()
                    
                    if self.invert_mask:
                        raw_mask = 1.0 - raw_mask
                        
                    # 3. Temporal Smoothing (EMA)
                    if prev_smoothed_mask is None:
                        smoothed_mask = raw_mask
                    else:
                        smoothed_mask = self.ema_alpha * raw_mask + (1.0 - self.ema_alpha) * prev_smoothed_mask
                    prev_smoothed_mask = smoothed_mask
                    
                    # 4. Feathering
                    feathered_mask = cv2.GaussianBlur(smoothed_mask, self.blur_kernel_size, 0)
                    
                    # 5. Validation Checks
                    coverage = np.mean(feathered_mask)
                    if coverage < self.min_coverage:
                        anomalies.append(f"Frame {frame_idx}: Low coverage ({coverage*100:.1f}%)")
                    elif coverage > self.max_coverage:
                        anomalies.append(f"Frame {frame_idx}: High coverage ({coverage*100:.1f}%)")
                        
                    if prev_coverage is not None:
                        variance = abs(coverage - prev_coverage)
                        if variance > self.max_variance:
                            anomalies.append(f"Frame {frame_idx}: High temporal variance ({variance*100:.1f}% jump)")
                    prev_coverage = coverage
                    
                    # 6. Compositing
                    mask_3d = np.stack((feathered_mask,) * 3, axis=-1)
                    composite = (frame * mask_3d + current_bg * (1.0 - mask_3d))
                    composite = np.ascontiguousarray(composite, dtype=np.uint8)
                    
                    out.write(composite)
                    frame_idx += 1
                    
                    if frame_idx % 30 == 0:
                        print(f"Processed {frame_idx}/{total_frames} frames...", flush=True)
            except Exception as e:
                import traceback
                print(f"Exception during processing at frame {frame_idx}:", flush=True)
                traceback.print_exc()
                        
        cap.release()
        if bg_cap:
            bg_cap.release()
        out.release()
        
        print(f"\n--- Validation Report for {os.path.basename(output_path)} ---", flush=True)
        if not anomalies:
            print("All clear! No coverage or temporal anomalies detected.", flush=True)
        else:
            print(f"Found {len(anomalies)} anomalies. Needs manual review:", flush=True)
            for a in anomalies[:10]: # Print first 10
                print(" - " + a, flush=True)
            if len(anomalies) > 10:
                print(f"   ... and {len(anomalies) - 10} more.", flush=True)
        print("--------------------------------------------------\n", flush=True)
        print("Finished process_video successfully!", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input PSL video")
    parser.add_argument("--bg", required=True, help="Background image or video")
    parser.add_argument("--output", required=True, help="Output composite video path")
    parser.add_argument("--ema", type=float, default=0.7, help="EMA Alpha")
    args = parser.parse_args()
    
    augmenter = PSLAugmenter(ema_alpha=args.ema)
    augmenter.process_video(args.input, args.bg, args.output)
