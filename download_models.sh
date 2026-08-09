#!/bin/bash
cd models/
echo "Downloading Face Landmarker..."
wget -nc -q --show-progress https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
echo "Downloading Hand Landmarker..."
wget -nc -q --show-progress https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
echo "Downloading Pose Landmarkers..."
wget -nc -q --show-progress https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
wget -nc -q --show-progress https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
wget -nc -q --show-progress https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
echo "Downloading Segmenters..."
wget -nc -q --show-progress https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite
wget -nc -q --show-progress https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite
wget -nc -q --show-progress https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite
echo "Downloading Face Stylizer..."
wget -nc -q --show-progress https://storage.googleapis.com/mediapipe-models/face_stylizer/blaze_face_stylizer/float16/latest/face_stylizer.task
