from imports import *

def generate_guiding_map(img):
    """
    Generates a multi-modality guidance map for cell segmentation.
    
    Works for RGB, grayscale, or multi-channel images.
    Steps:
    1. Convert to grayscale if needed
    2. Adaptive contrast enhancement
    3. Gaussian blur
    4. Thresholding (Otsu)
    5. Morphology (opening + closing)
    6. Normalization
    """
    # ---- Convert to grayscale ----
    if img.ndim == 3:
        if img.shape[2] == 3:
            img_gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
        else:
            # For multi-channel (like multi-fluorescence)
            img_gray = img.max(axis=-1)
    else:
        img_gray = img.astype(np.float32)
    
    # ---- Normalize and enhance contrast ----
    img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)
    img_contrast = exposure.equalize_adapthist(img_gray, clip_limit=0.03)  # Adaptive histogram equalization
    
    # ---- Gaussian blur ----
    img_blur = cv2.GaussianBlur(img_contrast, (3,3), 0)
    
    # ---- Thresholding (Otsu) ----
    try:
        t = threshold_otsu(img_blur)
    except ValueError:
        t = 0.5  # fallback if image is uniform
    mask = np.maximum(img_blur - t, 0)
    
    # ---- Morphology ----
    mask = opening(mask, footprint_rectangle((3,3)))
    mask = closing(mask, footprint_rectangle((3,3)))
    
    # ---- Min-max normalization ----
    mask_norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    
    return mask_norm[..., np.newaxis]  # (H,W,1)
