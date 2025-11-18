# YOLO11-OBB Training and Inference Documentation

## Project Overview

This project contains two main scripts designed for training and inference with the YOLO11-OBB (Oriented Bounding Box) model:

* `train_yolo11_obb_simple.py` — Simplified training script supporting training, curve plotting, and result evaluation
* `test_inference_only.py` — Script dedicated to model inference and visualization

Both scripts are intended for users who want a lightweight and easy-to-run pipeline for training YOLO11-OBB on custom datasets such as DOTA.


## Environment Setup

### 1. Create a Conda Virtual Environment

```bash
# Create a Conda environment (Python 3.9)
conda create -n yolo_env python=3.9 -y

# Activate the environment
conda activate yolo_env
```

### 2. Upgrade pip and Install Dependencies

```bash
# Install all required dependencies (includes PyTorch for CUDA 11.8)
pip install -r requirements.txt
```


## Dataset Preparation

Ensure the dataset directory follows the structure below:

```
data/
├── train/
│   ├── images/          # Training images
│   └── labels/          # Training labels (OBB format)
├── val/
│   ├── images/          # Validation images
│   └── labels/          # Validation labels
└── test/
    └── images/          # Test images
```

Make sure that the `dota_custom.yaml` configuration file is correctly set with dataset paths and the list of object categories.

## Usage

### Method 1: Full Training Pipeline

```bash
python train_yolo11_obb_simple.py
```

**Outputs:**

* `yolo11_final/dota_training/` — Training logs and model checkpoints
* `training_curves.png` — Loss and accuracy curves
* `detection_results/` — Visualization of detection outputs

---

### Method 2: Inference Only

```bash
python test_inference_only.py
```

**Features:**

* Supports specifying input images or randomly selecting from the test set
* Generates visualization results and detailed inference statistics
