# Enhancing 6D Object Pose Estimation

## Overview
This repository contains the codebase for a modular 6D object pose estimation pipeline evaluated on the LineMOD dataset.
The project builds upon a baseline pose estimation approach and introduces two modules to incorporate depth information in the learning process and enhance the predictions.

The repository includes training and evaluation code for individual components (detection, rotation, translation) as well as full end-to-end pipelines.


## Project Structure


The repository is organized as follows:
```
.
├── data/ # Dataset directory (automatically populated by scripts)
├── experiments/ # Training and testing entrypoints
│ ├── train/ # Training scripts for individual models
│ └── test/ # Evaluation scripts and full pipeline tests
├── notebooks/ # Example notebooks for inference and comparison
├── scripts/ # Utility scripts (data download, visualization, metrics)
├── src/ # Core library code
│ ├── datasets/ # Dataset definitions and loaders
│ ├── models/ # Model architectures
│ │ └── losses/ # Custom loss functions
│ └── utils/ # Geometry, camera models, visualization utilities
├── requirements.txt # Python dependencies
└── README.md
```

- `src/` contains all reusable components and is treated as the main Python package.
- `experiments/` contains the primary training and evaluation entrypoints.
- `scripts/` includes auxiliary tools for dataset preparation and analysis.
- `notebooks/` provides minimal, self-contained examples for running inference for the full pipeline.


## Installation

The project is implemented in Python and relies on PyTorch.

1. **Clone the repository**
```bash
git clone <repository-url>
cd 6d-pose-estimation
```

2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

3. Install dependencies

``` bash
pip install -r requirements.txt
```

All scripts are expected to be run from the repository root, so that the `src` package is correctly resolved by Python.

## Datasets

This project uses the **LineMOD** dataset as its benchmark.

### Downloading LineMOD
A helper script is provided to download and prepare the dataset:

```bash
python scripts/download_data.py
```

The dataset is downloaded as a compressed archive. After the download completes:

1. **Unzip the archive**
2. Keep the original folder name
3. Place the extracted directory inside `data/`

The expected default location is:

```
data/Linemod_preprocessed/
```

Most training and evaluation scripts assume this default path. If you choose a different location, you must explicitly pass the dataset path using the `--data_root` argument.

> **Recommendation:** After the first download, keep a local copy of `Linemod_preprocessed` and reuse it across experiments.

### YOLO Dataset Format
YOLO-based experiments require a different dataset structure. To generate it from LineMOD, use:

```bash
python scripts/export_dataset.py
```

This script creates a YOLO-compatible dataset at:

```
data/dataset_yolo/
```

If you move this directory, update the corresponding paths in the YOLO-related scripts using the `--data` or `--yolo_dataset` arguments:
- `train_yolo.py`
- `test_yolo.py`
- `test_baseline.py`
- `test_extension.py`

## Training
All training scripts are located in `experiments/train/` and are meant to be executed from the repository root.

Each training script saves:
- Weights of the last and the best model
- Logs on loss and metrics during training
- Sample images used during training

### YOLO Object Detector
Train the object detector on the YOLO-formatted LineMOD dataset:

```bash
python experiments/train/train_yolo.py \
    --model yolo11s.pt \
    --data data/dataset_yolo/data.yaml \
    --epochs 10 \
    --batch_size 16 \
    --out_dir train_yolo
```

### Rotation Estimation (RGB – ResNet)

Train the rotation network using RGB crops:

```bash
python experiments/train/train_resnet.py \
    --data_root data/Linemod_preprocessed \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --out_dir train_resnet
```

### Rotation Estimation (RGB-D Fusion)

Train the RGB-D fusion network for rotation estimation:

```bash
python experiments/train/train_rgbd.py \
    --data_root data/Linemod_preprocessed \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --out_dir train_rgbd
```

### Translation Estimation (Encoder–Decoder)

Train the encoder–decoder network for translation estimation:

```bash
python experiments/train/train_encoder_decoder.py \
    --data_root data/Linemod_preprocessed \
    --epochs 40 \
    --batch_size 64 \
    --lr 1e-3 \
    --out_dir train_encoder_decoder
```



## Evaluation

Evaluation scripts are located in experiments/test/ and provide quantitative evaluation of the different pipeline components.

Each evaluation script saves:
- Per-class metrics (CSV)
- Aggregate statistics
- Best / worst qualitative examples when applicable


### Baseline Pipeline (YOLO + RGB Rotation + Pinhole Translation)
```bash
python experiments/test/test_baseline.py \
    --yolo_model path/to/yolo_weights.pt \
    --resnet_model path/to/pose_resnet_weights.pth \
    --data_root data/Linemod_preprocessed \
    --yolo_dataset data/dataset_yolo \
    --out_dir test_baseline
```

### Full Extension Pipeline (YOLO + RGB-D Rotation + Encoder–Decoder Translation)
```bash
python experiments/test/test_extension.py \
    --yolo_model path/to/yolo_weights.pt \
    --rgbd_pose_model path/to/rgbd_pose_weights.pth \
    --enc_dec_model path/to/enc_dec_weights.pth \
    --data_root data/Linemod_preprocessed \
    --yolo_dataset data/dataset_yolo \
    --out_dir test_extension
```

### YOLO Detector Evaluation
```bash
python experiments/test/test_yolo.py \
    --model path/to/yolo_weights.pt \
    --data data/dataset_yolo/data.yaml
```

### Rotation-only evaluation (RGB – ResNet)

```bash
python experiments/test/test_resnet.py \
    --resnet_model path/to/pose_resnet_weights.pth \
    --data_root data/Linemod_preprocessed \
    --out_dir test_resnet
```


### Rotation Estimation (RGB-D Fusion)
```bash
python experiments/test/test_rgbd.py \
    --rgbd_pose_model path/to/rgbd_pose_weights.pth \
    --data_root data/Linemod_preprocessed \
    --out_dir test_rgbd
```

### Translation-only evaluation (Encoder–Decoder):

```bash
python experiments/test/test_enc_dec.py \
    --enc_dec_model path/to/enc_dec_weights.pth \
    --data_root data/Linemod_preprocessed \
    --out_dir test_enc_dec
```


