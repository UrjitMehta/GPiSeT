# Refined Swin-Unet Implementation for Cell Segmentation

This repository contains a modular implementation of a SwinUNet architecture for cell segmentation in microscopy images. The code is organized into separate components for clarity and reusability, allowing you to either:

- Run them as separate modules and orchestrate training/evaluation from your own main.py, or
    
- Combine everything into a single script for quick experiments.

The project_root/src/ directory contains the main components:

    imports.py – Centralized imports and global configurations (paths, libraries).
    
    dataloader.py – Dataset utilities for pairing images and labels, and creating generators.
    
    augmentation.py – Different augmentation techniques applied randomly (2–3 per image pair).
    
    model.py – Refined SwinUNet architecture.
    
    metrics.py – Loss functions (e.g., BCE + Dice).
    
    callbacks_checks.py – Custom metrics (Dice, F1, AP, AFNR).
    
    test.py – Testing and evaluation, including COCO-style metrics.
    
    train.py – Training loop with k-fold cross-validation.

Other files in the root directory:

    .gitignore – To ignore checkpoints, outputs, and logs.
    README.md – This documentation file.




## Configuration
- Paths (datasets, outputs, logs, models, etc) are defined in imports.py.

- Mixed precision (mixed_float16) can be enabled at the top of your training/testing scripts.

- Modify train_img_dir, train_lbl_dir, test_img_dir, test_lbl_dir in imports.py as needed.


## Usage

1. Install dependencies:
    pip install -r requirements.txt

2. Datasets:
- We combined a total of 5 publicly available datasets for training:
    1. [LIVECell Dataset](https://sartorius-research.github.io/LIVECell/)
    2. [Data Science Bowl 2018](https://bbbc.broadinstitute.org/BBBC038)
    3. [Cellpose](https://www.cellpose.org/)
    4. [Omnipose](https://osf.io/xmury/)
    5. [NeurIPS 2022 Cell Segmentation Challenge dataset](https://neurips22-cellseg.grand-challenge.org/dataset/)


3. Procedure:
    Import modules

    Prepare datasets
    
    Define the model
    
    Add metrics
    
    Set up callbacks, logging, and checks
    
    Train the model
    
    Test/evaluate the results
    
    (Optional) Use main.py to run everything in one go
   
## Notes
Each module is standalone; you can import functions/classes as needed:
    
    from data_loader import get_image_label_pairs, ImageLabelGenerator
    from model import swin_unet

If you prefer, you can create a main.py to link training and evaluation in one run.
Outputs (logs, curves, models, COCO JSONs, result images) are saved under outputs/, models/, summaries/, etc.
