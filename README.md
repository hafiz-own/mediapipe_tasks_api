# MediaPipe Vision Runner

A flexible, generalized engine for running Google MediaPipe Vision models on images, videos, folders, and live webcam feeds with practically zero boilerplate. 

This repository also includes specialized background augmentation tools designed for the Pakistan Sign Language (PSL) Dataset.

## Features

* **Universal Abstraction Engine (`utils/pipeline.py`)**: The `MediaPipeRunner` class dynamically inspects MediaPipe tasks (Landmarkers, Segmenters, Classifiers), configures the correct options, and handles media I/O entirely behind the scenes.
* **Auto-Mode Detection**: The runner automatically detects whether an input is a single image, video file, or an entire folder of mixed media. 
* **Live Webcam Support**: Supports asynchronous processing for real-time live webcam feeds.
* **Specialized Draw Callbacks**: Extracted out-of-the-box OpenCV rendering modules for Pose, Face, Hands, Holistic, and Segmentation masks.
* **Hyperparameter Injection**: Passes down model-specific CLI arguments directly to the MediaPipe options builder.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd mediapipe
   ```

2. **Set up the virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Download the pre-trained models:**
   Run the downloader script to fetch the `.task` files from Google:
   ```bash
   bash download_models.sh
   ```

## Usage

You can find the wrapper scripts in the `scripts/` directory. They all support the same consistent CLI structure.

### Supported Models
* `run_holistic.py` (Full Body, Face, Hands)
* `run_face.py` (Face Landmarker)
* `run_hands.py` (Hand Landmarker)
* `run_pose.py` (Pose Landmarker)
* `run_segmentation.py` (Background/Foreground Segmentation)

### Examples

**1. Process a Single Image or Video**
```bash
python scripts/run_holistic.py --input data/sample.mp4
```
*(The engine automatically identifies `.mp4` as a video and processes frame-by-frame.)*

**2. Batch Process an Entire Directory**
```bash
python scripts/run_hands.py --input data/my_dataset/
```
*(The engine will process all valid media files in the folder and save them to `output/my_dataset_annotated`.)*

**3. Run Live Webcam Tracking**
```bash
python scripts/run_pose.py --live
```

**4. Explicit Mode (Optional)**
```bash
python scripts/run_segmentation.py --mode image --input data/sample.jpg
```
*(Supported modes: `image`, `video`, `live`, `auto`. Defaults to `auto`.)*

---

## PSL Background Augmentation Pipeline

This repository includes a specialized background augmentation module designed for a sign language recognition dataset. It uses MediaPipe's Image Segmenter to extract human subjects and composite them onto different backgrounds.

### Features
* **Temporal Smoothing (EMA)** to reduce mask flickering between video frames.
* **Gaussian Feathering** for natural edge blending.
* **Video & Image Backgrounds**: Supports compositing subjects onto static images, chroma key colors (e.g., `solid_green`), or moving video backgrounds.
* **Anomaly Detection**: Tracks unexpected jumps in mask coverage to flag potentially corrupt frames.

### Usage
```bash
python scripts/psl_augmenter.py \
    --input data/signer_video.mp4 \
    --bg data/busy_market.jpg \
    --output output/augmented_signer.mp4
```

## Repository Structure
* `scripts/`: Standalone execution wrappers for different MediaPipe models.
* `utils/pipeline.py`: The core execution engine.
* `utils/draw_*.py`: Rendering utilities for annotated outputs.
* `models/`: Directory for downloaded `.task` models.
* `output/`: Default output directory for processed files.
