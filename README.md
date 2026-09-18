# GPiSeT: Guidance Fused Pixel-level Cell Segmentation Framework with Transformer Backbone

The Overall Architecture:
<img width="1000" height="1007" alt="generic_whole_arch" src="https://github.com/user-attachments/assets/855e662a-8a15-4baf-9d92-73b32c3c71f8" />


This repository contains a modular implementation of a guidance fused SwinUNet architecture for cell segmentation in microscopy images. The code is organized into separate components for clarity and reusability, allowing you to either:

- Run them as separate modules and orchestrate training/evaluation from your own main.py, or
    
- Combine everything into a single script for quick experiments.

The project_root/src/ directory contains the main components:

    imports.py – Centralized imports and global configurations (paths, libraries).
    
    dataloader.py – Dataset utilities for pairing images and labels, and creating generators.
    
    augmentation.py – Different augmentation techniques applied randomly (2–3 per image pair).
    
    model.py – Refined SwinUNet architecture.
    
    metrics.py – Loss functions (e.g., BCE + Dice).
    
    guidance_generation/ – Scripts for generating guidance maps using either:
        heuristic.py – Heuristic-based guidance generation.
        mini_cnn.py – CNN-derived guidance generation. The core data loading
                      and training pipeline is currently configured for
                      Mini-CNN-generated guidance maps.
    
    callbacks_checks.py – Custom metrics (Dice, F1, AP, AFNR).
    
    test.py – Testing and evaluation, including COCO-style metrics.
    
    train.py – Training loop with k-fold cross-validation.

Other files in the root directory:

    .gitignore – To ignore checkpoints, outputs, and logs.
    README.md – This documentation file.
    download_model.py – Download the pretrained GPiSeT models from Hugging Face.




## Configuration
- Paths (datasets, outputs, logs, models, etc) are defined in imports.py.

- Mixed precision (mixed_float16) can be enabled at the top of your training/testing scripts.

- Modify train_img_dir, train_lbl_dir, test_img_dir, test_lbl_dir in imports.py as needed.

## Guidance Generation

The framework supports two approaches for generating guidance maps, implemented
in the `guidance_generation/` directory:

- `heuristic.py` – Generates guidance maps using a heuristic-based approach.

  Heuristic-based Guidance Map Generation Pipeline:
  <img width="5423" height="1096" alt="HEU-pipeline_colored" src="https://github.com/user-attachments/assets/affde5d4-219c-4da7-b4f3-5bc9f718ca76" />

- `mini_cnn.py` – Learns and generates guidance maps using a lightweight U-Net-Styled CNN-based approach.

  CNN-Derived Guidance Map Generation Pipeline:
  <img width="5655" height="3546" alt="CNN-pipeline_colored" src="https://github.com/user-attachments/assets/a931f1ae-fd12-4483-a2bc-89e105b2a6c2" />

Either approach can be used to generate the guidance maps required by the
guidance-fused SwinUNet framework. The core data loading and training pipeline
is currently configured for guidance maps generated using `mini_cnn.py`.
If the heuristic-based approach is preferred, the corresponding guidance
generation script and paths can be adapted accordingly.

## Usage

1. Install dependencies:
    pip install -r requirements.txt

2. Pretrained Model:

- A pretrained version of the proposed GPiSeT models are available on Hugging Face. The pretrained models were selected from the 5-fold cross-validation experiments based on the best      validation Dice score.

- You can use the pretrained models in either of the following ways:

- **Download using the provided script:**

    ```bash
    python download_model.py
    ```

    This will automatically download the pretrained `GPiSeT-*.keras` and make it available locally for testing or inference on your own dataset.

- **Download directly from Hugging Face:**

    The pretrained `GPiSeT-*.keras` can also be downloaded directly from the [Hugging Face Model Repository](https://huggingface.co/urjit006/GPiSeT) and used according to your requirements.

- **Fine-tune from the GPiSeT architecture:**

    The GPiSeT architecture can also be initialized from `model.py` and fine-tuned or trained on a custom dataset. Configure the required dataset and output paths in `imports.py` before training.

3. Datasets:
- We combined a total of 5 publicly available datasets for training:
    1. [LIVECell Dataset](https://sartorius-research.github.io/LIVECell/)
    2. [Data Science Bowl 2018](https://bbbc.broadinstitute.org/BBBC038)
    3. [Cellpose](https://www.cellpose.org/)
    4. [NeurIPS 2022 Cell Segmentation Challenge dataset](https://neurips22-cellseg.grand-challenge.org/dataset/)


4. Procedure:
    Import modules

    Prepare datasets

    Prepare Guidance Maps
    
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

## Citation

If you use this repository or find the work helpful, please cite the paper:

```bibtex
@inproceedings{um2026gpiset,
  title={GPiSeT: Guidance Fused Pixel-Level Cell Segmentation Framework with Transformer Backbone},
  author={Mehta, Urjit and Bhalodiya, Jayendra},
  booktitle={48th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
  year={2026},
  url={https://github.com/UrjitMehta/GPiSeT}
}
```

If you use the model in your work, please cite the repo:
```bibtex
@misc{gpiset_model,
	author       = {Mehta, Urjit and Bhalodiya, Jayendra},
	title        = { GPiSeT},
	year         = 2026,
	url          = { https://huggingface.co/urjit006/GPiSeT },
	doi          = { 10.57967/hf/10499 },
	publisher    = { Hugging Face }
}
```
