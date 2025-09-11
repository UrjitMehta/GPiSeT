# Cell-Segmentation-using-Refined-SwinUNet-Implementation

This repository contains a modular implementation of a SwinUNet architecture for cell segmentation in microscopy images. The code is organized into separate components for clarity and reusability, allowing you to either:

Run them as separate modules and orchestrate training/evaluation from your own main.py, or

Combine everything into a single script for quick experiments.

project_root/
├──src
    ├── imports.py              # Centralized imports and global configs (paths, libs)
    ├── dataloader.py           # Dataset utilities (pairing images/labels, generators)
    ├── augmentation.py         # Different Augmentation techniques(randomly picked in pair of 2-3)
    ├── model.py                # Refined SwinUNet architecture
    ├── metrics.py              # Loss functions (BCE + Dice, etc.)
    ├── callbacks_checks.py     # Custom metrics (Dice, F1, AP, AFNR)
    ├── test.py                 # Custom training callbacks (checkpoints, COCO eval, etc.)
    ├── train.py                # Training with k-fold validation method

├── .gitignore          # Ignore checkpoints, outputs, logs
└── README.md           # This file


## Configuration
Paths (datasets, outputs, logs, models, etc) are defined in imports.py.

Mixed precision (mixed_float16) can be enabled at the top of your training/testing scripts.

Modify train_img_dir, train_lbl_dir, test_img_dir, test_lbl_dir in imports.py as needed.


## Usage

1. Install dependencies:
    pip install -r requirements.txt

2. Datasets:
- Used total of 5 publicaly available datasets.
- Combined while training: 
    1. [LIVECell Dataset](https://sartorius-research.github.io/LIVECell/)
    2. [Data Science Bowl 2018](https://bbbc.broadinstitute.org/BBBC038)
    3. [Cellpose](https://www.cellpose.org/)
    4. [Omnipose](https://osf.io/xmury/)
    5. [NeurIPS 2022 Cell Segmentation Challenge dataset](https://neurips22-cellseg.grand-challenge.org/dataset/)


3. Procedure:
    - imports
    - data preparation
    - model definition
    - metrics adding 
    - callbacks, logging and checks
    - training
    - testing
    - main(optional)

## Notes
Each module is standalone; you can import functions/classes as needed:
    from data_loader import get_image_label_pairs, ImageLabelGenerator
    from model import swin_unet

If you prefer, you can create a main.py to link training and evaluation in one run.
Outputs (logs, curves, models, COCO JSONs, result images) are saved under outputs/, models/, summaries/, etc.