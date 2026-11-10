"""Replace the background of a PSL signer video using MediaPipe's selfie segmenter (CPU).

Example:
    uv run python scripts/psl_augmenter.py --input data/signer.mp4 --bg data/market.jpg --output output/aug.mp4
    --bg also accepts a video, or the keywords solid_green / solid_gray.
"""
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

SOLID_COLOURS = {"solid_green": (0, 255, 0), "solid_gray": (128, 128, 128)}  # BGR
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


class PSLAugmenter:
    def __init__(self,
                 ema_alpha=0.7,
                 blur_kernel_size=(21, 21),
                 min_coverage=0.02,
                 max_coverage=0.80,
                 max_variance=0.10,
                 invert_mask=False,  # confidence_masks[0] is category 0 (person)
                 model_path=MODEL_PATH):
        self.ema_alpha = ema_alpha
        self.blur_kernel_size = blur_kernel_size
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage
        self.max_variance = max_variance
        self.invert_mask = invert_mask

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}\nRun ./download_models.sh first.")
        base_options = python.BaseOptions(model_asset_path=model_path, delegate=python.BaseOptions.Delegate.CPU)
        self.options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_category_mask=False,
            output_confidence_masks=True
        )

    def _prepare_bg_frame(self, bg_src, target_shape):
        """Centre-crops and resizes the background to exactly the target (H, W)."""
        target_h, target_w = target_shape[:2]
        bg_h, bg_w = bg_src.shape[:2]
        target_aspect = target_w / target_h
        bg_aspect = bg_w / bg_h

        if bg_aspect > target_aspect:  # background is wider: crop the sides
            new_w = int(bg_h * target_aspect)
            start_x = (bg_w - new_w) // 2
            cropped = bg_src[:, start_x:start_x + new_w]
        else:  # background is taller: crop top/bottom
            new_h = int(bg_w / target_aspect)
            start_y = (bg_h - new_h) // 2
            cropped = bg_src[start_y:start_y + new_h, :]

        return cv2.resize(cropped, (target_w, target_h))

    def process_video(self, input_video_path, bg_path, output_path):
        print(f"Starting process_video with input={input_video_path}, bg={bg_path} (delegate=CPU, mediapipe {mp.__version__})")
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open input video: {input_video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            print("Warning: input reports no FPS; assuming 30.")
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Load the background (fail early with a clear message instead of crashing mid-way)
        bg_cap = None
        static_bg = None
        if bg_path in SOLID_COLOURS:
            static_bg = np.full((height, width, 3), SOLID_COLOURS[bg_path], dtype=np.uint8)
        elif bg_path.lower().endswith(VIDEO_EXTS):
            bg_cap = cv2.VideoCapture(bg_path)
            if not bg_cap.isOpened():
                cap.release()
                raise FileNotFoundError(f"Could not open background video: {bg_path}")
        else:
            img = cv2.imread(bg_path)
            if img is None:
                cap.release()
                raise FileNotFoundError(f"Could not read background image: {bg_path} "
                                        f"(or use one of: {', '.join(SOLID_COLOURS)})")
            static_bg = self._prepare_bg_frame(img, (height, width, 3))

        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        prev_smoothed_mask = None
        prev_coverage = None
        anomalies = []
        frame_idx = 0

        with vision.ImageSegmenter.create_from_options(self.options) as segmenter:
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 1. Background frame
                    if bg_cap is not None:
                        bg_ret, bg_frame = bg_cap.read()
                        if not bg_ret:  # loop the background video if it's shorter
                            bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            _, bg_frame = bg_cap.read()
                        current_bg = self._prepare_bg_frame(bg_frame, frame.shape)
                    else:
                        current_bg = static_bg

                    # 2. Segmentation
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    timestamp_ms = int(round(frame_idx * 1000.0 / fps))
                    result = segmenter.segment_for_video(mp_image, timestamp_ms)
                    raw_mask = result.confidence_masks[0].numpy_view().squeeze().astype(np.float32)
                    if self.invert_mask:
                        raw_mask = 1.0 - raw_mask

                    # 3. Temporal smoothing (EMA)
                    if prev_smoothed_mask is None:
                        smoothed_mask = raw_mask
                    else:
                        smoothed_mask = self.ema_alpha * raw_mask + (1.0 - self.ema_alpha) * prev_smoothed_mask
                    prev_smoothed_mask = smoothed_mask

                    # 4. Feathering
                    feathered_mask = cv2.GaussianBlur(smoothed_mask, self.blur_kernel_size, 0)

                    # 5. Validation checks
                    coverage = float(np.mean(feathered_mask))
                    if coverage < self.min_coverage:
                        anomalies.append(f"Frame {frame_idx}: Low coverage ({coverage*100:.1f}%)")
                    elif coverage > self.max_coverage:
                        anomalies.append(f"Frame {frame_idx}: High coverage ({coverage*100:.1f}%)")
                    if prev_coverage is not None:
                        variance = abs(coverage - prev_coverage)
                        if variance > self.max_variance:
                            anomalies.append(f"Frame {frame_idx}: High temporal variance ({variance*100:.1f}% jump)")
                    prev_coverage = coverage

                    # 6. Compositing in float32 (float64 was ~2x slower for no benefit)
                    mask_3d = feathered_mask[..., None]
                    composite = frame.astype(np.float32) * mask_3d + current_bg.astype(np.float32) * (1.0 - mask_3d)
                    out.write(np.clip(composite, 0, 255).astype(np.uint8))
                    frame_idx += 1

                    if frame_idx % 30 == 0:
                        print(f"Processed {frame_idx}/{total_frames} frames...", flush=True)
            except Exception:
                import traceback
                print(f"Exception during processing at frame {frame_idx}:", flush=True)
                traceback.print_exc()
                raise
            finally:
                cap.release()
                if bg_cap is not None:
                    bg_cap.release()
                out.release()

        print(f"\n--- Validation Report for {os.path.basename(output_path)} ---", flush=True)
        if not anomalies:
            print("All clear! No coverage or temporal anomalies detected.", flush=True)
        else:
            print(f"Found {len(anomalies)} anomalies. Needs manual review:", flush=True)
            for a in anomalies[:10]:
                print(" - " + a, flush=True)
            if len(anomalies) > 10:
                print(f"   ... and {len(anomalies) - 10} more.", flush=True)
        print("--------------------------------------------------\n", flush=True)
        print(f"Finished: {frame_idx} frames written to {output_path}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Replace the background of a PSL signer video.")
    parser.add_argument("--input", required=True, help="Input PSL video")
    parser.add_argument("--bg", required=True, help="Background image, video, or solid_green / solid_gray")
    parser.add_argument("--output", required=True, help="Output composite video path")
    parser.add_argument("--ema", type=float, default=0.7, help="EMA alpha for mask smoothing (default 0.7)")
    parser.add_argument("--feather", type=int, default=21, help="Odd Gaussian kernel size for edge feathering (default 21)")
    parser.add_argument("--invert_mask", action="store_true", help="Invert the mask if the person/background come out swapped")
    parser.add_argument("--model", default=MODEL_PATH, help="Segmentation model (.tflite)")
    args = parser.parse_args()
    if args.feather % 2 == 0:
        parser.error("--feather must be an odd number")

    augmenter = PSLAugmenter(ema_alpha=args.ema, blur_kernel_size=(args.feather, args.feather),
                             invert_mask=args.invert_mask, model_path=args.model)
    try:
        augmenter.process_video(args.input, args.bg, args.output)
    except FileNotFoundError as e:
        sys.exit(f"Error: {e}")
