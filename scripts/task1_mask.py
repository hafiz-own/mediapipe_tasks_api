"""Inspect the selfie segmenter's confidence masks on the first frame of a video.

Example:
    uv run python scripts/task1_mask.py --input /path/to/clip.mp4
"""
import argparse
import os
import cv2
import numpy as np
import sys
# Vision only: stop MediaPipe from loading its audio library (sounddevice -> PortAudio -> PipeWire).
# That import can hang forever, and sets a realtime CPU limit of 0. See docs/bug_resolutions.md, section 5.
sys.modules.setdefault("sounddevice", None)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/selfie_segmenter.tflite")


def main():
    parser = argparse.ArgumentParser(description="Save the segmenter's raw confidence masks for one video frame.")
    parser.add_argument("--input", required=True, help="Input video")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to inspect (default 0)")
    parser.add_argument("--model", default=MODEL_PATH, help="Segmentation model (.tflite)")
    parser.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "output"), help="Where to save the masks")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.input)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise SystemExit(f"Could not read frame {args.frame} from {args.input}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=args.model, delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=False,
        output_confidence_masks=True,
    )

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        confidence_masks = segmenter.segment(mp_image).confidence_masks
        print(f"Number of confidence masks returned: {len(confidence_masks)}")

        # Save every mask, since which index is the person depends on the model.
        for i, mask in enumerate(confidence_masks):
            mask_data = mask.numpy_view()
            print(f"Mask {i}: shape={mask_data.shape}, dtype={mask_data.dtype}, "
                  f"min={mask_data.min():.3f}, max={mask_data.max():.3f}")
            out = os.path.join(args.output_dir, f"task1_mask_index_{i}.jpg")
            cv2.imwrite(out, (mask_data * 255).astype(np.uint8))  # float [0,1] -> uint8 [0,255]
            print(f"Saved {out}")


if __name__ == "__main__":
    main()
