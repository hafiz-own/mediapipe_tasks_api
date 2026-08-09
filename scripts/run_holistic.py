import argparse
import os
import sys
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from utils import draw_holistic_landmarks
from utils.pipeline import MediaPipeRunner

DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models/holistic_landmarker.task")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "output")

def main():
    parser = argparse.ArgumentParser(
        description="Holistic Landmarker Engine",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples:
  1. Run on a single image (auto-detects image mode):
     python scripts/run_holistic.py --input data/image.jpg

  2. Run on an entire folder of mixed images/videos (auto mode):
     python scripts/run_holistic.py --input data/my_dataset

  3. Run live webcam feed:
     python scripts/run_holistic.py --mode live
"""
    )
    
    parser.add_argument("-i", "--input", type=str, default="",
                        help="Path to the input image, video file, or folder")
    parser.add_argument("--live", action="store_true",
                        help="Run live webcam feed instead of processing an input")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to the holistic_landmarker.task file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT,
                        help="Directory to save the generated outputs")
    
    # Model Specific Hyperparameters
    parser.add_argument("--min_face_detection_confidence", type=float, default=0.5,
                        help="Confidence threshold for face detection")
    parser.add_argument("--min_face_suppression_threshold", type=float, default=0.3,
                        help="Confidence threshold for face suppression")
    parser.add_argument("--min_face_landmarks_confidence", type=float, default=0.5,
                        help="Confidence threshold for face landmarks")
    parser.add_argument("--min_pose_detection_confidence", type=float, default=0.5,
                        help="Confidence threshold for pose detection")
    parser.add_argument("--min_pose_suppression_threshold", type=float, default=0.3,
                        help="Confidence threshold for pose suppression")
    parser.add_argument("--min_pose_landmarks_confidence", type=float, default=0.5,
                        help="Confidence threshold for pose landmarks")
    parser.add_argument("--min_hand_landmarks_confidence", type=float, default=0.5,
                        help="Confidence threshold for hand landmarks")
                        
    args = parser.parse_args()
    
    if not args.live and not args.input:
        parser.error("You must provide either --input or --live.")
        
    mode = "live" if args.live else "auto"
        
    runner = MediaPipeRunner(
        task_class=vision.HolisticLandmarker,
        options_class=vision.HolisticLandmarkerOptions,
        model_path=args.model,
        drawing_callback=draw_holistic_landmarks,
        # Dynamically pass down all specific arguments
        min_face_detection_confidence=args.min_face_detection_confidence,
        min_face_suppression_threshold=args.min_face_suppression_threshold,
        min_face_landmarks_confidence=args.min_face_landmarks_confidence,
        min_pose_detection_confidence=args.min_pose_detection_confidence,
        min_pose_suppression_threshold=args.min_pose_suppression_threshold,
        min_pose_landmarks_confidence=args.min_pose_landmarks_confidence,
        min_hand_landmarks_confidence=args.min_hand_landmarks_confidence
    )
    
    runner.run(input_path=args.input, output_dir=args.output_dir, mode=mode)

if __name__ == "__main__":
    main()
