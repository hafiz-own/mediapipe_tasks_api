import argparse
import os
import sys
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from utils import draw_face_result
from utils.pipeline import MediaPipeRunner

DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models/face_landmarker.task")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "output")

def main():
    parser = argparse.ArgumentParser(
        description="Face Landmarker Engine",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples:
  1. Run on an entire folder of mixed images/videos (auto mode):
     python scripts/run_face.py --input data/my_dataset

  2. Run live webcam feed:
     python scripts/run_face.py --mode live
"""
    )
    
    parser.add_argument("-i", "--input", type=str, default="",
                        help="Path to the input image, video file, or folder")
    parser.add_argument("--live", action="store_true",
                        help="Run live webcam feed instead of processing an input")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to the face_landmarker.task file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT,
                        help="Directory to save the generated outputs")
    
    parser.add_argument("--mode", type=str, choices=["image", "video", "live", "auto"], default="auto",
                        help="Processing mode (default: auto). Auto detects based on file extension.")
    
    # Model Specific Hyperparameters
    parser.add_argument("--num_faces", type=int, default=1,
                        help="Maximum number of faces to detect")
    parser.add_argument("--min_face_detection_confidence", type=float, default=0.5,
                        help="Confidence threshold for face detection")
    parser.add_argument("--min_face_presence_confidence", type=float, default=0.5,
                        help="Confidence threshold for face presence")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5,
                        help="Confidence threshold for face tracking")
    parser.add_argument("--output_face_blendshapes", action="store_true",
                        help="Output face blendshapes (default False)")
    parser.add_argument("--output_facial_transformation_matrixes", action="store_true",
                        help="Output facial transformation matrixes (default False)")
                        
    args = parser.parse_args()
    
    if not args.live and not args.input and args.mode != "live":
        parser.error("You must provide either --input or --live.")
        
    mode = "live" if args.live else "auto"
        
    runner = MediaPipeRunner(
        task_class=vision.FaceLandmarker,
        options_class=vision.FaceLandmarkerOptions,
        model_path=args.model,
        drawing_callback=draw_face_result,
        # Dynamically pass down all specific arguments
        num_faces=args.num_faces,
        min_face_detection_confidence=args.min_face_detection_confidence,
        min_face_presence_confidence=args.min_face_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        output_face_blendshapes=args.output_face_blendshapes,
        output_facial_transformation_matrixes=args.output_facial_transformation_matrixes
    )
    
    runner.run(input_path=args.input, output_dir=args.output_dir, mode=mode)

if __name__ == "__main__":
    main()
