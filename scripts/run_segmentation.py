import argparse
import os
from functools import partial
import sys
# Vision only: stop MediaPipe from loading its audio library (sounddevice -> PortAudio -> PipeWire).
# That import can hang forever, and sets a realtime CPU limit of 0. See docs/bug_resolutions.md, section 5.
sys.modules.setdefault("sounddevice", None)
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from utils import draw_segmentation_result
from utils.pipeline import MediaPipeRunner

DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models/selfie_segmenter.tflite")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "output")


def main():
    parser = argparse.ArgumentParser(
        description="Image Segmenter Engine",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples:
  1. Run on a single image or video (auto-detects the mode):
     uv run python scripts/run_segmentation.py --input data/sample.mp4

  2. Run on an entire folder of mixed images/videos:
     uv run python scripts/run_segmentation.py --input data/my_dataset

  3. Run live webcam feed:
     uv run python scripts/run_segmentation.py --live
"""
    )

    parser.add_argument("-i", "--input", type=str, default="",
                        help="Path to the input image, video file, or folder")
    parser.add_argument("--live", action="store_true",
                        help="Run live webcam feed instead of processing an input")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to the segmentation model (selfie_segmenter, hair_segmenter, deeplab_v3)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT,
                        help="Directory to save the generated outputs")
    parser.add_argument("--mode", type=str, choices=["image", "video", "live", "auto"], default="auto",
                        help="Processing mode (default: auto). Auto detects based on file extension.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam index for live mode (default 0)")
    parser.add_argument("--visual_mode", type=str, default="blur", choices=["blur", "desaturate", "overlay"],
                        help="How to visualize the segmentation (blur, desaturate, overlay)")

    # Model Specific Hyperparameters
    parser.add_argument("--output_category_mask", action=argparse.BooleanOptionalAction, default=True,
                        help="Output the category mask (needed for drawing). Use --no-output_category_mask to turn off")
    parser.add_argument("--output_confidence_masks", action="store_true",
                        help="Also output confidence masks (default False)")
    args = parser.parse_args()

    if not args.live and not args.input and args.mode != "live":
        parser.error("You must provide either --input or set --mode live / --live.")

    mode = "live" if args.live else args.mode

    if not args.output_category_mask and not args.output_confidence_masks:
        parser.error("MediaPipe needs at least one mask: drop --no-output_category_mask or add --output_confidence_masks.")
    if not args.output_category_mask:
        print("Warning: --no-output_category_mask set; outputs will be saved without any segmentation drawn.")

    # Inject the visual mode into the drawing callback
    visualizer = partial(draw_segmentation_result, visual_mode=args.visual_mode)

    runner = MediaPipeRunner(
        task_class=vision.ImageSegmenter,
        options_class=vision.ImageSegmenterOptions,
        model_path=args.model,
        drawing_callback=visualizer,
        camera_index=args.camera,
        # Model-specific options, passed straight to MediaPipe
        output_category_mask=args.output_category_mask,
        output_confidence_masks=args.output_confidence_masks
    )

    runner.run(input_path=args.input, output_dir=args.output_dir, mode=mode)


if __name__ == "__main__":
    main()
