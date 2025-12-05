# Waste Detection Using YOLO

This project implements a real-time waste detection system using YOLO (You Only Look Once) deep learning models. The system can detect different types of construction and demolition waste and is designed to be extended for real-time classification using a laptop camera.

---

## Features

- Detects multiple classes of construction waste.
- Dataset preprocessing and YOLO annotation conversion included.
- YOLOv8 model trained on a custom dataset.
- Real-time detection script ready to use with a webcam.
- Supports further deployment using Flask or Streamlit.

---

## Requirements

- Python 3.10+
- Libraries:
  - [Ultralytics](https://pypi.org/project/ultralytics/)
  - OpenCV
  - NumPy
  - Matplotlib
  - Flask or Streamlit for deployment

---

## Setup Instructions

**Clone the repository:**

git clone https://github.com/yourusername/waste-detection.git
cd waste-detection

## Install Dependencies

pip install ultralytics opencv-python numpy matplotlib

## Run real-time detection (laptop camera required):

python realtime_detection.py

## Dataset Preparation

XML annotations are converted to YOLO .txt format using convert_annotation.py.
Dataset split into training, validation, and testing sets.
data.yaml defines paths and class names for YOLO.

## Notes

Ensure the correct Python version is used (3.10 recommended).
Webcam access is required for real-time detection.
Deployment via Flask or Streamlit can be added for user-friendly interface.
