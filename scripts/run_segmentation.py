import argparse
import os
import sys
from functools import partial
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
  1. Run with background blur (default):
     python scripts/run_segmentation.py --input data/image.jpg

  2. Run with black-and-white background:
     python scripts/run_segmentation.py --input data/image.jpg --visual_mode desaturate
"""
    )
    
    parser.add_argument("-i", "--input", type=str, default="",
                        help="Path to the input image, video file, or folder")
    parser.add_argument("--live", action="store_true",
                        help="Run live webcam feed instead of processing an input")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to the segmentation model file (e.g., selfie_segmenter, deeplabv3)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT,
                        help="Directory to save the generated outputs")
    
    # Visualization Mode
    parser.add_argument("--visual_mode", type=str, default="blur", choices=["blur", "desaturate", "overlay"],
                        help="How to visualize the segmentation (blur, desaturate, overlay)")
    
    parser.add_argument("--mode", type=str, choices=["image", "video", "live", "auto"], default="auto",
                        help="Processing mode (default: auto). Auto detects based on file extension.")
    
    # Model Specific Hyperparameters
    parser.add_argument("--output_category_mask", action="store_true", default=True,
                        help="Output category mask (must be True for visualization)")
    parser.add_argument("--output_confidence_masks", action="store_true", default=False,
                        help="Whether to output confidence masks")
                        
    args = parser.parse_args()
    
    if not args.live and not args.input and args.mode != "live":
        parser.error("You must provide either --input or set --mode live / --live.")
        
    mode = "live" if args.live else args.mode
        
    # Dynamically inject the visual mode into the drawing callback
    visualizer = partial(draw_segmentation_result, visual_mode=args.visual_mode)
        
    runner = MediaPipeRunner(
        task_class=vision.ImageSegmenter,
        options_class=vision.ImageSegmenterOptions,
        model_path=args.model,
        drawing_callback=visualizer,
        # Dynamically pass down MediaPipe specific arguments
        output_category_mask=args.output_category_mask,
        output_confidence_masks=args.output_confidence_masks
    )
    
    runner.run(input_path=args.input, output_dir=args.output_dir, mode=mode)

if __name__ == "__main__":
    main()
