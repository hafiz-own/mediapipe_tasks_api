import argparse
import os
import sys
# Vision only: stop MediaPipe from loading its audio library (sounddevice -> PortAudio -> PipeWire).
# That import can hang forever, and sets a realtime CPU limit of 0. See docs/bug_resolutions.md, section 5.
sys.modules.setdefault("sounddevice", None)
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from utils import draw_pose_result
from utils.pipeline import MediaPipeRunner

DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models/pose_landmarker_lite.task")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "output")


def main():
    parser = argparse.ArgumentParser(
        description="Pose Landmarker Engine",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples:
  1. Run on a single image or video (auto-detects the mode):
     uv run python scripts/run_pose.py --input data/sample.mp4

  2. Run on an entire folder of mixed images/videos:
     uv run python scripts/run_pose.py --input data/my_dataset

  3. Run live webcam feed:
     uv run python scripts/run_pose.py --live
"""
    )

    parser.add_argument("-i", "--input", type=str, default="",
                        help="Path to the input image, video file, or folder")
    parser.add_argument("--live", action="store_true",
                        help="Run live webcam feed instead of processing an input")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to the pose_landmarker model file (lite, full or heavy)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT,
                        help="Directory to save the generated outputs")
    parser.add_argument("--mode", type=str, choices=["image", "video", "live", "auto"], default="auto",
                        help="Processing mode (default: auto). Auto detects based on file extension.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam index for live mode (default 0)")

    # Model Specific Hyperparameters
    parser.add_argument("--num_poses", type=int, default=1,
                        help="Maximum number of poses to detect")
    parser.add_argument("--min_pose_detection_confidence", type=float, default=0.5,
                        help="Confidence threshold for pose detection")
    parser.add_argument("--min_pose_presence_confidence", type=float, default=0.5,
                        help="Confidence threshold for pose presence")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5,
                        help="Confidence threshold for tracking")
    parser.add_argument("--output_segmentation_masks", action="store_true",
                        help="Output segmentation masks (default False)")
    args = parser.parse_args()

    if not args.live and not args.input and args.mode != "live":
        parser.error("You must provide either --input or set --mode live / --live.")

    mode = "live" if args.live else args.mode

    runner = MediaPipeRunner(
        task_class=vision.PoseLandmarker,
        options_class=vision.PoseLandmarkerOptions,
        model_path=args.model,
        drawing_callback=draw_pose_result,
        camera_index=args.camera,
        # Model-specific options, passed straight to MediaPipe
        num_poses=args.num_poses,
        min_pose_detection_confidence=args.min_pose_detection_confidence,
        min_pose_presence_confidence=args.min_pose_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        output_segmentation_masks=args.output_segmentation_masks
    )

    runner.run(input_path=args.input, output_dir=args.output_dir, mode=mode)


if __name__ == "__main__":
    main()
