import os
import cv2
import time
import mimetypes
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class MediaPipeRunner:
    def __init__(self, 
                 task_class, 
                 options_class, 
                 model_path, 
                 drawing_callback,
                 delegate=python.BaseOptions.Delegate.CPU,
                 **model_kwargs):
        
        self.task_class = task_class
        self.options_class = options_class
        self.model_path = os.path.abspath(model_path)
        self.drawing_callback = drawing_callback
        self.delegate = delegate
        self.model_kwargs = model_kwargs 
        
        self._latest_result = None
        self._new_result = False

    def _get_options(self, running_mode, result_callback=None):
        base_options = python.BaseOptions(
            model_asset_path=self.model_path,
            delegate=self.delegate
        )
        kwargs = self.model_kwargs.copy()
        kwargs['base_options'] = base_options
        kwargs['running_mode'] = running_mode
        if result_callback:
            kwargs['result_callback'] = result_callback
            
        return self.options_class(**kwargs)

    def _execute_task(self, model, mode, *args):
        """Dynamically finds if the model uses .detect(), .segment(), or .classify()"""
        method_map = {
            vision.RunningMode.IMAGE: ['detect', 'segment', 'classify', 'recognize'],
            vision.RunningMode.VIDEO: ['detect_for_video', 'segment_for_video', 'classify_for_video', 'recognize_for_video'],
            vision.RunningMode.LIVE_STREAM: ['detect_async', 'segment_async', 'classify_async', 'recognize_async']
        }
        
        for method_name in method_map[mode]:
            if hasattr(model, method_name):
                func = getattr(model, method_name)
                return func(*args)
                
        raise ValueError(f"Task class {self.task_class.__name__} has no known execution method for {mode}.")

    def _update_result_callback(self, result, image, timestamp_ms):
        self._latest_result = result
        self._new_result = True

    def _run_image(self, input_path, output_dir):
        print(f"Running Image Mode on {input_path}")
        options = self._get_options(vision.RunningMode.IMAGE)
        
        with self.task_class.create_from_options(options) as model:
            mp_image = mp.Image.create_from_file(input_path)
            
            result = self._execute_task(model, vision.RunningMode.IMAGE, mp_image)
            
            annotated_image = self.drawing_callback(mp_image.numpy_view(), result)
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            filename = os.path.basename(input_path)
            if 'Segmenter' in self.task_class.__name__:
                prefix = 'segmentation'
            else:
                prefix = self.task_class.__name__.replace('Landmarker', '').lower()
            out_path = os.path.join(output_dir, f"{prefix}_annotated_{filename}")
            cv2.imwrite(out_path, annotated_image_bgr)
            print(f"Saved annotated image to: {out_path}")

    def _run_video(self, input_path, output_dir):
        print(f"Running Video Mode on {input_path}")
        options = self._get_options(vision.RunningMode.VIDEO)
        
        with self.task_class.create_from_options(options) as model:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"Error opening video file {input_path}")
                return
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            filename = os.path.basename(input_path)
            if 'Segmenter' in self.task_class.__name__:
                prefix = 'segmentation'
            else:
                prefix = self.task_class.__name__.replace('Landmarker', '').lower()
            out_path = os.path.join(output_dir, f"{prefix}_annotated_{filename}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                if frame_timestamp_ms == 0 and cap.get(cv2.CAP_PROP_POS_FRAMES) > 1:
                     frame_timestamp_ms = int((cap.get(cv2.CAP_PROP_POS_FRAMES) / fps) * 1000)
                
                result = self._execute_task(model, vision.RunningMode.VIDEO, mp_image, frame_timestamp_ms)
                
                annotated_image = self.drawing_callback(frame_rgb, result)
                annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                
                out.write(annotated_image_bgr)

            cap.release()
            out.release()
            print(f"Saved annotated video to: {out_path}")

    def _run_live(self):
        print("Running Live Stream Mode. Press 'q' to exit.")
        options = self._get_options(vision.RunningMode.LIVE_STREAM, result_callback=self._update_result_callback)
        
        with self.task_class.create_from_options(options) as model:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open webcam.")
                return

            window_name = f"{self.task_class.__name__} - Live"

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Error reading from webcam.")
                    break
                    
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                frame_timestamp_ms = int(time.time() * 1000)
                
                self._execute_task(model, vision.RunningMode.LIVE_STREAM, mp_image, frame_timestamp_ms)
                
                if self._new_result and self._latest_result:
                    annotated_image = self.drawing_callback(frame_rgb, self._latest_result)
                    display_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                    self._new_result = False
                else:
                    display_image = frame
                    
                cv2.imshow(window_name, display_image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

    def process_file(self, input_path, output_dir, mode="auto", is_batch=False):
        mime_type, _ = mimetypes.guess_type(input_path)
        if not mime_type:
            if is_batch:
                print(f"Skipping {input_path}: Unknown file type.")
                return
            else:
                raise ValueError(f"Unknown file type for {input_path}")

        is_image = mime_type.startswith('image')
        is_video = mime_type.startswith('video')

        if not is_image and not is_video:
            if is_batch:
                print(f"Skipping {input_path}: Not a valid image or video (detected {mime_type}).")
                return
            else:
                raise ValueError(f"Invalid file format for {input_path}. Must be image or video (detected {mime_type}).")

        if mode == "image" and not is_image:
            print(f"Warning: Mode is set to 'image' but '{input_path}' is a video. Skipping.")
            return
        elif mode == "video" and not is_video:
            print(f"Warning: Mode is set to 'video' but '{input_path}' is an image. Skipping.")
            return

        target_mode = mode
        if mode == "auto":
            target_mode = "image" if is_image else "video"

        if target_mode == "image":
            self._run_image(input_path, output_dir)
        elif target_mode == "video":
            self._run_video(input_path, output_dir)

    def run(self, input_path=None, output_dir="output", mode="auto"):
        if mode == "live":
            self._run_live()
            return

        if not input_path:
            raise ValueError("Input path is required for image, video, and auto modes.")

        abs_input_path = os.path.abspath(input_path)
        if not os.path.exists(abs_input_path):
            raise FileNotFoundError(f"Input path '{abs_input_path}' does not exist.")

        output_base_dir = os.path.abspath(output_dir)

        if os.path.isdir(abs_input_path):
            folder_name = os.path.basename(abs_input_path.rstrip(os.sep))
            out_dir = os.path.join(output_base_dir, f"{folder_name}_annotated")
            os.makedirs(out_dir, exist_ok=True)
            print(f"Batch processing folder: {abs_input_path}")
            print(f"Outputs will be saved to: {out_dir}")
            for root, _, files in os.walk(abs_input_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    self.process_file(file_path, out_dir, mode, is_batch=True)
        else:
            os.makedirs(output_base_dir, exist_ok=True)
            self.process_file(abs_input_path, output_base_dir, mode, is_batch=False)
