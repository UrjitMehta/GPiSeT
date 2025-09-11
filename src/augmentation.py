import numpy as np
import cv2

# ========== AUGMENTATIONS ==========

def zoom_image(img, scale_range=(0.25, 1.5)):
    scale = np.random.uniform(*scale_range)
    h, w = img.shape[:2]
    c = 1 if img.ndim == 2 else img.shape[2]
    new_h, new_w = int(h * scale), int(w * scale)
    img_zoomed = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    if img_zoomed.ndim == 2 and c == 1:
        img_zoomed = np.expand_dims(img_zoomed, axis=-1)
    if scale < 1.0:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        img_padded = np.zeros((h, w, c), dtype=img.dtype)
        img_padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_zoomed
        return img_padded
    else:
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        img_cropped = img_zoomed[start_h:start_h+h, start_w:start_w+w]
        return np.expand_dims(img_cropped, axis=-1) if img_cropped.ndim == 2 and c == 1 else img_cropped

def rotate_image(img, k=None):
    if k is None:
        k = np.random.choice([1, 2, 3])
    return np.rot90(img, k=k)

def cell_aware_intensity_scale(img, scale_range=(1.0, 1.7)):
    return np.clip(img * np.random.uniform(*scale_range), 0, 1)

def add_gaussian_noise(img, mean=0, std=0.1):
    noise = np.random.normal(mean, std, img.shape)
    return np.clip(img + noise, 0, 1)

def contrast_adjustment(img, gamma_range=(0, 2)):
    return np.clip(np.power(img, np.random.uniform(*gamma_range)), 0, 1)

def gaussian_smoothing(img, sigma=1.0):
    ksize = int(2*np.ceil(3*sigma)+1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def histogram_shift(img):
    x1, y1 = np.random.uniform(0.3, 0.7), np.random.uniform(0.3, 0.7)
    def piecewise(x): return (y1 / x1) * x if x < x1 else y1 + ((1 - y1) / (1 - x1)) * (x - x1)
    return np.clip(np.vectorize(piecewise)(img), 0, 1)

def gaussian_sharpening(img, sigma=0.5, alpha_range=(10, 30)):
    blurred = gaussian_smoothing(img, sigma)
    return np.clip(img + np.random.uniform(*alpha_range) * (img - blurred), 0, 1)