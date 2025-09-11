# ===================================
# Master Imports for SwinUNet Project
# Save this as imports.py
# ===================================

# ---- Standard Library
import os
import gc
import math
import json
import warnings

# ---- Core Scientific Stack
import numpy as np
import matplotlib.pyplot as plt

# ---- Image I/O & Processing
import cv2
import tifffile as tiff
from skimage import measure
from skimage.transform import resize as sk_resize
from skimage.filters import threshold_otsu

# ---- Machine Learning
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

# ---- Scikit-learn for splits
from sklearn.model_selection import KFold

# ---- COCO Evaluation
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ---- Keras Utils
from tensorflow.keras.utils import Sequence, to_categorical
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# ---- Any custom utilities from your project
# (Uncomment and adjust paths if you keep them in separate .py files)
# from data_loader import get_image_label_pairs, ImageLabelGenerator
# from losses import bce_dice_loss, dice_coef
# from metrics import F1Score, ap_metric, AFNR
from callbacks_checks import (
    EpochCheckpoint,
    SavePredictionsCallback,
    OutputSanityCheckCallback,
    COCOEvalCallback
)
from  metrics import *
from model import *
from dataloader import *
from augmentation import *
from train import get_image_label_pairs, get_logger, all_val_losss, all_val_dice_scores


# Paths
train_img_dir = 'train_images'
train_lbl_dir = 'train_all_TIFF_labels'
test_img_dir = 'test_images'
test_lbl_dir = 'test_all_TIFF_labels'

model_dir = 'swinUnet'
loss_dir = 'swinUnet_metrics_plots'
output_dir = 'swinUnet_outputs'


log_dir = "logs/fits/swinUnet_tensorflow_logs"

log_path = 'evaluation_logs'

summary_dir = "swinUnet_model_summaries"

os.makedirs(summary_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(loss_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)