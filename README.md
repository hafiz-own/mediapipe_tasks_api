# MediaPipe Vision Runner

Runs Google MediaPipe Tasks vision models (holistic, face, hands, pose, segmentation) on images, videos, whole folders, or a live webcam. Also includes a background augmenter for the PSL dataset. **CPU only**, with the same MediaPipe version as `signbase-psl`, so landmarks match in both repos.

**Contents**
1. Setup (uv)
2. Running the models
3. PSL background augmenter
4. Why CPU only, and why these versions
5. Repository layout

## 1. Setup (uv)

```bash
cd /data/Study/Current/FYP-1/mediapipe_tasks_api
uv sync                  # creates .venv with Python 3.11 and the pinned packages
./download_models.sh     # only if models/ is missing files
```

## 2. Running the models

Every runner has the same options: `--input` (image, video or folder), `--live` (webcam), `--mode image|video|live|auto` (default `auto`), `--output_dir` (default `output/`), `--model`, `--camera N`, plus model-specific thresholds (`--help` lists them).

| Script | Model |
|---|---|
| `scripts/run_holistic.py` | full body + face + hands |
| `scripts/run_face.py` | face mesh |
| `scripts/run_hands.py` | hands |
| `scripts/run_pose.py` | pose (lite by default; `--model models/pose_landmarker_full.task`) |
| `scripts/run_segmentation.py` | person/background (`--visual_mode blur|desaturate|overlay`) |

```bash
uv run python scripts/run_holistic.py --input data/sample.mp4     # one video
uv run python scripts/run_hands.py --input data/my_dataset/       # whole folder (sub-folders kept)
uv run python scripts/run_pose.py --live                          # webcam; press q or close the window to quit
```

Each run prints what it used: model, `delegate=CPU`, MediaPipe version.

Live mode does not mirror anything (not the frame the model sees, not the window), so left and right hands are correct.

Holistic output draws the signer's **left hand in orange** and **right hand in light blue**, with a small key in the top-left corner. Left and right mean the signer's own hands, not the sides of the image.

Folder runs keep going if a file fails, and finish with a summary (processed / skipped / failed).

## 3. PSL background augmenter

Cuts the signer out with the selfie segmenter and puts them on a new background, with EMA smoothing against flicker, feathered edges, and a report of suspicious frames.

```bash
uv run python scripts/psl_augmenter.py --input data/signer.mp4 --bg data/market.jpg --output output/aug.mp4
```

`--bg` takes an image, a video (it loops), or `solid_green` / `solid_gray`. Options: `--ema 0.7`, `--feather 21` (odd), `--invert_mask`.

## 4. Why CPU only, and why these versions

- **CPU only.** The GPU path was tested on 2026-11-10 (MX550 and Intel iGPU). Holistic crashes on the GPU with MediaPipe 0.10.35. With 1.2.2 it ran, but it was slower than CPU, used more CPU, and shifted landmarks (pose about 10 px). Details are in `docs/bug_resolutions.md`.
- **mediapipe 0.10.35 + numpy 1.26.4.** These match the `signbase-psl` training cache. Changing them changes landmarks, so do it only together with a cache rebuild.
- **No audio.** Every script blocks MediaPipe's audio library before importing MediaPipe, because it can hang at start-up (`docs/bug_resolutions.md`, section 5).
- **opencv-contrib-python only.** Don't also install `opencv-python`; both provide `cv2` and break each other.

## 5. Repository layout

- `scripts/`: the runners above and the augmenter.
- `utils/pipeline.py`: `MediaPipeRunner`, the shared engine (modes, file I/O, webcam).
- `utils/draw_*.py`: drawing helpers (scaled for 4K, handle holistic's flat landmark lists).
- `models/`: the downloaded `.task` / `.tflite` files (`download_models.sh`).
- `docs/bug_resolutions.md`: problems hit so far and how they were solved.
- `output/`: results (git-ignored).
