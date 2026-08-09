# Bug Resolutions & Learnings

This document chronicles the major bugs encountered and resolved during the integration of the MediaPipe Tasks API `HolisticLandmarker` module into our Python environment.

## 1. Out-of-Memory (Exit Code 137) during `create_from_options`
**Symptoms:** 
- The script would exit with `Exit Code 137` (SIGKILL) on Linux/macOS before processing any images.
- Even on machines with 8GB+ of RAM, the Python process would be abruptly terminated by the OS Out-Of-Memory killer.
- The minimal reproducible script would either silently exit with `Exit Code 0` without triggering a Python exception (even with `faulthandler` enabled), or be forcefully killed.

**Root Cause:**
- **Incompatibility between MediaPipe 1.0.0 and Numpy 2.x.** The latest bleeding-edge releases of MediaPipe (e.g. 1.0.0) were installed alongside `numpy==2.5.1`. The C++ tensor bindings used in `create_from_options` failed to properly interpret the new Numpy 2.x ABI arrays, triggering a massive, uncontrollable memory leak and/or a silent C++ abort.

**Resolution:**
- Downgraded MediaPipe to the latest stable minor branch (`mediapipe==0.10.35`) and `numpy` to the `1.26.x` branch.
- **Reference:** See `requirements.txt` which locks `mediapipe==0.10.35` and `numpy==1.26.4`.

## 2. GPU Delegation Segmentation Fault
**Symptoms:**
- Supplying `Delegate.GPU` to the `BaseOptions` configuration for the Tasks API `HolisticLandmarker` caused a direct segmentation fault (SIGSEGV) upon model initialization.

**Root Cause:**
- There is a known bug in the current MediaPipe Tasks API implementations for heavy composite models (like Holistic) running on desktop GPU delegates (OpenGL/OpenCL). The model graph architectures are currently highly unstable on desktop GPUs via the Tasks API C++ backend.

**Resolution:**
- Forced the use of the CPU delegate via `BaseOptions.Delegate.CPU`. The XNNPACK CPU delegate is highly optimized and provides near real-time performance even for the heavy holistic model without risking crashes.

## 3. Drawing Utilities `TypeError: 'NormalizedLandmark' object is not iterable`
**Symptoms:**
- The model successfully generated inference results, but the script crashed at the drawing phase when using `drawing_utils.draw_landmarks`.
- The crash occurred because the loop tried to iterate over a single `NormalizedLandmark`.

**Root Cause:**
- The official Google examples (e.g., `PoseLandmarker`) return lists of lists for landmarks (`List[List[NormalizedLandmark]]`) because they support multi-person detection. However, `HolisticLandmarkerResult` currently assumes a single subject and directly returns a flat list (`List[NormalizedLandmark]`) for face, pose, and hand components.
- Directly copying the `for loop` structure from the Pose Landmarker example caused an iterable mismatch on the flat list.

**Resolution:**
- Removed the `for` loops in the `draw_landmarks_on_image` function and passed the `detection_result.face_landmarks`, `detection_result.pose_landmarks`, etc., directly to `landmark_list` as flat lists.

## 4. Landmark Blobs on High-Resolution Images
**Symptoms:**
- On high-resolution input images (e.g., 4K), dynamically scaling the `DrawingSpec` thickened the connection lines nicely, but equally scaled the `circle_radius` of the landmarks. This resulted in the landmark dots expanding into massive, overlapping blobs that obscured the subject's face and hands.

**Root Cause:**
- Applying the same scale factor indiscriminately to both the `circle_radius` (dots) and `thickness` (lines) breaks visual proportions, as `drawing_utils` uses `thickness` for the connection lines but `circle_radius` for the landmark nodes.

**Resolution:**
- Decoupled the scaling logic into `scale_connection_spec` (scales heavily relative to width, e.g., `width / 300`) and `scale_landmark_spec` (scales minimally relative to width, e.g., `width / 1000`). This keeps connection lines bold and visible on 4K imagery while keeping the coordinate dots small, precise, and unobtrusive.
