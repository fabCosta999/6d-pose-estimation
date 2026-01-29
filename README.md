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
python -m scripts.download_data
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
python -m scripts.export_dataset
```

This script creates a YOLO-compatible dataset at:

```
data/dataset_yolo/
```

If you move this directory, update the corresponding paths in the YOLO-related scripts using the `--data` or `--yolo_dataset` arguments:
- `train_yolo.py`
- `test_yolo.py`
- `test_pipeline_baseline.py`
- `test_pipeline_extension.py`

## Training
All training scripts are located in `experiments/train/` and are meant to be executed from the repository root.

Each training script saves:
- Weights of the last and the best model
- Logs on loss and metrics during training
- Sample images used during training

### YOLO Object Detector
Train the object detector on the YOLO-formatted LineMOD dataset:

```bash
python -m experiments.train.train_yolo \
    --yolo_weights yolo11s.pt \
    --data data/dataset_yolo/data.yaml \
    --epochs 10 \
    --batch_size 16 \
    --out_dir train_yolo
```

### Rotation Estimation (RGB – ResNet)

Train the rotation network using RGB crops:

```bash
python -m experiments.train.train_resnet \
    --data_root data/Linemod_preprocessed \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --out_dir train_resnet
```

### Rotation Estimation (RGB-D Fusion)

Train the RGB-D fusion network for rotation estimation:

```bash
python -m experiments.train.train_rotation_extension \
    --data_root data/Linemod_preprocessed \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --out_dir train_rotation_extension
```

### Translation Estimation (Encoder–Decoder)

Train the encoder–decoder network for translation estimation:

```bash
python -m experiments.train.train_translation_extension \
    --data_root data/Linemod_preprocessed \
    --epochs 40 \
    --batch_size 64 \
    --lr 1e-3 \
    --out_dir train_translation_extension
```



## Evaluation

Evaluation scripts are located in experiments/test/ and provide quantitative evaluation of the different pipeline components.

Each evaluation script saves:
- Per-class metrics (CSV)
- Aggregate statistics
- Best / worst qualitative examples when applicable


### Baseline Pipeline (YOLO + RGB Rotation + Pinhole Translation)
```bash
python -m experiments.test.test_pipeline_baseline \
    --yolo_weights path/to/yolo_weights.pt \
    --resnet_weights path/to/resnet_weights.pth \
    --data_root data/Linemod_preprocessed \
    --yolo_dataset data/dataset_yolo \
    --out_dir test_pipeline_baseline
```

### Full Extension Pipeline (YOLO + RGB-D Rotation + Encoder–Decoder Translation)
```bash
python -m experiments.test.test_pipeline_extension \
    --yolo_weights path/to/yolo_weights.pt \
    --rot_ext_weights path/to/rot_ext_weights.pth \
    --trans_ext_weights path/to/trans_ext_weights.pth \
    --data_root data/Linemod_preprocessed \
    --yolo_dataset data/dataset_yolo \
    --out_dir test_pipeline_extension
```

### YOLO Detector Evaluation
```bash
python -m experiments.test.test_yolo \
    --yolo_weights path/to/yolo_weights.pt \
    --data data/dataset_yolo/data.yaml
```

### Rotation-only evaluation (RGB – ResNet)

```bash
python -m experiments.test.test_resnet \
    --resnet_weights path/to/resnet_weights.pth \
    --data_root data/Linemod_preprocessed \
    --out_dir test_resnet
```


### Rotation Estimation (RGB-D Fusion)
```bash
python -m experiments.test.test_rotation_extension \
    --rot_ext_weights path/to/rot_ext_weights.pth \
    --data_root data/Linemod_preprocessed \
    --out_dir test_rotation_extension
```

### Translation-only evaluation (Encoder–Decoder):

```bash
python -m experiments.test.test_translation_extension \
    --trans_ext_weights path/to/trans_ext_weights.pth \
    --data_root data/Linemod_preprocessed \
    --out_dir test_translation_extension
```


