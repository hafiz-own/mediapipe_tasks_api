import sys
# Vision only: stop MediaPipe from loading its audio library (sounddevice -> PortAudio -> PipeWire).
# That import can hang forever, and sets a realtime CPU limit of 0. See docs/bug_resolutions.md, section 5.
sys.modules.setdefault("sounddevice", None)
import os
import cv2
import time
import threading
import mimetypes
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# This repo runs MediaPipe on CPU only (XNNPACK). GPU was tested on 2026-11-10 and
# gave no speed gain for Holistic while shifting landmarks; see docs/bug_resolutions.md.
DELEGATE = python.BaseOptions.Delegate.CPU


class MediaPipeRunner:
    def __init__(self,
                 task_class,
                 options_class,
                 model_path,
                 drawing_callback,
                 camera_index=0,
                 **model_kwargs):

        self.task_class = task_class
        self.options_class = options_class
        self.model_path = os.path.abspath(model_path)
        self.drawing_callback = drawing_callback
        # Live mode does NOT mirror: neither the frame the model sees nor the window.
        # (Mirroring would swap left/right hands; if added later, do it for both consistently.)
        self.camera_index = camera_index
        self.model_kwargs = model_kwargs

        self._lock = threading.Lock()
        self._latest = None  # (result, rgb_frame) from the live-stream callback

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\nRun ./download_models.sh first.")

    # ── setup ────────────────────────────────────────────────────────────────

    def _get_options(self, running_mode, result_callback=None):
        base_options = python.BaseOptions(
            model_asset_path=self.model_path,
            delegate=DELEGATE,
        )
        kwargs = self.model_kwargs.copy()
        kwargs['base_options'] = base_options
        kwargs['running_mode'] = running_mode
        if result_callback:
            kwargs['result_callback'] = result_callback
        return self.options_class(**kwargs)

    def _banner(self, mode):
        print(f"[{self.task_class.__name__}] mode={mode} | model={os.path.basename(self.model_path)} "
              f"| delegate=CPU | mediapipe {mp.__version__}")

    def _execute_task(self, model, mode, *args):
        """Dynamically finds if the model uses .detect(), .segment(), .classify() or .recognize()"""
        method_map = {
            vision.RunningMode.IMAGE: ['detect', 'segment', 'classify', 'recognize'],
            vision.RunningMode.VIDEO: ['detect_for_video', 'segment_for_video', 'classify_for_video', 'recognize_for_video'],
            vision.RunningMode.LIVE_STREAM: ['detect_async', 'segment_async', 'classify_async', 'recognize_async']
        }
        for method_name in method_map[mode]:
            if hasattr(model, method_name):
                return getattr(model, method_name)(*args)
        raise ValueError(f"Task class {self.task_class.__name__} has no known execution method for {mode}.")

    def _prefix(self):
        if 'Segmenter' in self.task_class.__name__:
            return 'segmentation'
        return self.task_class.__name__.replace('Landmarker', '').lower()

    # ── image ────────────────────────────────────────────────────────────────

    def _run_image(self, input_path, output_dir):
        print(f"Running Image Mode on {input_path}")
        options = self._get_options(vision.RunningMode.IMAGE)

        with self.task_class.create_from_options(options) as model:
            mp_image = mp.Image.create_from_file(input_path)
            result = self._execute_task(model, vision.RunningMode.IMAGE, mp_image)

            annotated_image = self.drawing_callback(mp_image.numpy_view(), result)
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

            out_path = os.path.join(output_dir, f"{self._prefix()}_annotated_{os.path.basename(input_path)}")
            cv2.imwrite(out_path, annotated_image_bgr)
            print(f"Saved annotated image to: {out_path}")

    # ── video ────────────────────────────────────────────────────────────────

    def _run_video(self, input_path, output_dir):
        print(f"Running Video Mode on {input_path}")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error opening video file {input_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            print(f"Warning: {input_path} reports no FPS; assuming 30.")
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        name = os.path.splitext(os.path.basename(input_path))[0]
        out_path = os.path.join(output_dir, f"{self._prefix()}_annotated_{name}.mp4")
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        options = self._get_options(vision.RunningMode.VIDEO)
        frame_idx = 0
        with self.task_class.create_from_options(options) as model:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # Timestamps from the frame index: always strictly increasing, which
                # MediaPipe requires (CAP_PROP_POS_MSEC can repeat or read 0).
                frame_timestamp_ms = int(round(frame_idx * 1000.0 / fps))
                result = self._execute_task(model, vision.RunningMode.VIDEO, mp_image, frame_timestamp_ms)

                annotated_image = self.drawing_callback(frame_rgb, result)
                out.write(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                frame_idx += 1

        cap.release()
        out.release()
        print(f"Saved annotated video ({frame_idx} frames) to: {out_path}")

    # ── live ─────────────────────────────────────────────────────────────────

    def _update_result_callback(self, result, output_image, timestamp_ms):
        # Keep the frame MediaPipe actually processed, so landmarks are drawn on the
        # frame they belong to (not on a newer webcam frame).
        frame = output_image.numpy_view().copy()
        with self._lock:
            self._latest = (result, frame)

    def _run_live(self):
        print("Running Live Stream Mode. Press 'q' to exit.")
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"Error: Could not open webcam {self.camera_index}.")
            return

        options = self._get_options(vision.RunningMode.LIVE_STREAM, result_callback=self._update_result_callback)
        window_name = f"{self.task_class.__name__} - Live"
        last_ts = -1

        with self.task_class.create_from_options(options) as model:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error reading from webcam.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # Monotonic clock (never jumps back), forced strictly increasing.
                ts = int(time.monotonic() * 1000)
                if ts <= last_ts:
                    ts = last_ts + 1
                last_ts = ts
                self._execute_task(model, vision.RunningMode.LIVE_STREAM, mp_image, ts)

                with self._lock:
                    latest = self._latest
                if latest is not None:
                    result, processed_rgb = latest
                    display = cv2.cvtColor(self.drawing_callback(processed_rgb, result), cv2.COLOR_RGB2BGR)
                else:
                    display = frame

                cv2.imshow(window_name, display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    # ── dispatch ─────────────────────────────────────────────────────────────

    def process_file(self, input_path, output_dir, mode="auto", is_batch=False):
        mime_type, _ = mimetypes.guess_type(input_path)
        if not mime_type:
            if is_batch:
                print(f"Skipping {input_path}: Unknown file type.")
                return
            raise ValueError(f"Unknown file type for {input_path}")

        is_image = mime_type.startswith('image')
        is_video = mime_type.startswith('video')

        if not is_image and not is_video:
            if is_batch:
                print(f"Skipping {input_path}: Not a valid image or video (detected {mime_type}).")
                return
            raise ValueError(f"Invalid file format for {input_path}. Must be image or video (detected {mime_type}).")

        if mode == "image" and not is_image:
            print(f"Warning: Mode is set to 'image' but '{input_path}' is a video. Skipping.")
            return
        if mode == "video" and not is_video:
            print(f"Warning: Mode is set to 'video' but '{input_path}' is an image. Skipping.")
            return

        os.makedirs(output_dir, exist_ok=True)
        target_mode = mode if mode != "auto" else ("image" if is_image else "video")
        if target_mode == "image":
            self._run_image(input_path, output_dir)
        else:
            self._run_video(input_path, output_dir)

    def run(self, input_path=None, output_dir="output", mode="auto"):
        self._banner(mode)
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
            print(f"Batch processing folder: {abs_input_path}")
            print(f"Outputs will be saved to: {out_dir}")
            for root, _, files in os.walk(abs_input_path):
                # Mirror sub-folders so files with the same name in different
                # folders don't overwrite each other.
                rel = os.path.relpath(root, abs_input_path)
                target_dir = out_dir if rel == "." else os.path.join(out_dir, rel)
                for file in sorted(files):
                    self.process_file(os.path.join(root, file), target_dir, mode, is_batch=True)
        else:
            self.process_file(abs_input_path, output_base_dir, mode, is_batch=False)
