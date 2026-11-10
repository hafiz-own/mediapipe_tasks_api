# Bug Resolutions & Learnings

Major bugs hit while using the MediaPipe Tasks API in Python, and how they were resolved.

**Contents**
1. Exit code 137 during `create_from_options` (updated 2026-11-10: real cause found)
2. GPU delegate segfault / failure (updated 2026-11-10)
3. Drawing: `'NormalizedLandmark' object is not iterable`
4. Landmark blobs on high-resolution images
5. ★ `import mediapipe` loads an audio library: random hangs, and MediaPipe 1.x gets killed
6. NVIDIA GPU: `eglGetDisplay()` returns no display on a hybrid-graphics laptop
7. ★ GPU vs CPU benchmark: why this repo stays on CPU

## 1. Exit code 137 (SIGKILL) during `create_from_options`
**Symptoms:**
- The script exited with `Exit Code 137` (SIGKILL) before processing any images, even with plenty of free RAM.
- No Python exception, even with `faulthandler` enabled.

**Original diagnosis (Aug 2026):** blamed on MediaPipe 1.0.0 + NumPy 2.x causing a memory leak. Downgrading to `mediapipe==0.10.35` and `numpy==1.26.4` made it go away.

**Update (2026-11-10):** the kernel log shows no out-of-memory kill, and MediaPipe 1.2.2 dies the same way with NumPy 1.26.4. So NumPy was most likely not the cause. The real cause is section 5. The pins stay because they match the `signbase-psl` training cache.

## 2. GPU delegate failure
**Symptoms:** `Delegate.GPU` crashed (SIGSEGV) at model initialization on the old laptop.

**Root cause:** the old laptop had no usable GPU. MediaPipe's GPU path on Linux is OpenGL ES through EGL, not CUDA, so it had nothing to run on.

**Update (2026-11-10, laptop with MX550 + Intel Iris Xe):**
- With MediaPipe 0.10.35, hands, pose, face and the segmenters run on the GPU.
- Holistic still fails on any GPU with `Batch size mismatch, expected 1 but got 52` (its face-blendshapes model). MediaPipe 1.2.0 fixed that ("run face blendshapes on CPU when using the GPU delegate").
- See section 7 for whether it's worth using.

**Resolution:** CPU delegate (XNNPACK) everywhere. `utils/pipeline.py` has `DELEGATE = CPU`.

## 3. Drawing Utilities `TypeError: 'NormalizedLandmark' object is not iterable`
**Symptoms:** inference worked, but drawing crashed in `drawing_utils.draw_landmarks`.

**Root cause:** the Pose/Face/Hand landmarkers return `List[List[NormalizedLandmark]]` (multi-person). `HolisticLandmarkerResult` assumes one person and returns flat lists.

**Resolution:** the `draw_*` helpers check `hasattr(landmarks[0], 'x')` and wrap flat lists, so both shapes work.

## 4. Landmark Blobs on High-Resolution Images
**Symptoms:** on 4K images, scaling `DrawingSpec` made the landmark dots into large blobs.

**Root cause:** the same scale factor was applied to `circle_radius` (dots) and `thickness` (lines).

**Resolution:** separate `scale_connection_spec` (about `width / 300`) and `scale_landmark_spec` (about `width / 1000`) in `utils/draw_core.py`.

## 5. `import mediapipe` loads an audio library
**Symptoms (two of them):**
- **Random hang at start-up (0.10.35, found 2026-11-11).** A script freezes before printing anything, using 0% CPU. It happened in about 1 of 5 runs from a shell without the desktop session variables. The core dump shows it stuck in `Pa_Initialize` → ALSA → `libasound_module_pcm_pipewire` → `pw_thread_loop_stop`.
- **MediaPipe 1.x killed (found 2026-11-10).** MediaPipe 1.2.2 exits with 137 on CPU and GPU when run from a terminal in the desktop session. The same script works with the session variables removed. `xdg-desktop-portal` logs `Realtime error: Could not map tid ...` at the moment of death.

**Root cause:** `import mediapipe` imports `sounddevice` (`tasks/python/audio/core/audio_record.py`), which starts PortAudio → PipeWire. When the session bus is reachable (`XDG_RUNTIME_DIR` or `DBUS_SESSION_BUS_ADDRESS` set), PipeWire sets `RLIMIT_RTTIME` to 0 and asks the portal to make a thread realtime. MediaPipe 1.x then gets SIGKILLed by the kernel. 0.10.35 also gets `RLIMIT_RTTIME = 0`, but so far survives it.

**Resolution (applied in every script and `utils/pipeline.py`):** block the audio import before importing MediaPipe. MediaPipe wraps that import in `try`, so vision tasks are unaffected:
```python
import sys
sys.modules.setdefault("sounddevice", None)
import mediapipe as mp
```
Anything else that imports MediaPipe (for example `signbase-psl`) can hit the same hang and needs the same two lines.

## 6. NVIDIA GPU: `eglGetDisplay()` returned no display
**Symptoms:** with `__EGL_VENDOR_LIBRARY_FILENAMES=.../10_nvidia.json`, every GPU task fails with `display != EGL_NO_DISPLAY` (`gl_context_egl.cc:77`). `prime-run` doesn't help.

**Root cause:** all outputs of this laptop are wired to the Intel GPU, so the Wayland session runs on Intel. MediaPipe asks EGL for the default display, which is the Wayland session, and NVIDIA's EGL can't attach to it.

**Resolution (if GPU is ever needed):** run with NVIDIA as the only EGL vendor and without the display variables:
```bash
env -u DISPLAY -u WAYLAND_DISPLAY __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json uv run python ...
```
Don't use a vendor list with Mesa as a fallback (like the `use-gpu` alias): if NVIDIA fails, MediaPipe silently lands on the Intel GPU.

## 7. GPU vs CPU benchmark (2026-11-10)
10 dataset clips, 1,387 frames, 1920×1080, holistic in VIDEO mode:

| Run | ms per frame | CPU ms per frame |
|---|---|---|
| 0.10.35 CPU | 27 | 87 |
| 1.2.2 CPU | 25–32 | 70–91 |
| 1.2.2 GPU (MX550) | 38 | 96 |

Landmark differences (mean, at 1920 px width):
- 0.10.35 vs 1.2.2 CPU: about 1–2 px, hand found in one but not the other in about 2% of frames.
- CPU vs GPU: pose about 10 px, hands about 3.5 px, hand presence differs in about 2.5–3% of frames.

**Decision:** stay on CPU and 0.10.35. The GPU is slower, doesn't take load off the CPU, and changes landmarks relative to the training cache.
