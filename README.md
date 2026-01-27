# Enhancing 6D Object Pose Estimation

## Overview
This repository contains the codebase for a modular 6D object pose estimation pipeline evaluated on the LineMOD dataset.
The project builds upon a baseline pose estimation approach and introduces two modules to incorporate depth information in the learning process and enhance the preditions.

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
- `notebooks/` provides minimal, self-contained examples for running inference and comparing methods.


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
LineMOD: come scaricarlo / dove metterlo.

## Training
Comandi per trainare:
- baseline
- encoder-decoder
- full extension

## Evaluation
Comandi per testare e riprodurre i numeri del paper.

## Qualitative Results
(1–2 immagini o riferimento alle figure del paper)

## Notes
Differenze di split, DenseFusion come reference, ecc.
