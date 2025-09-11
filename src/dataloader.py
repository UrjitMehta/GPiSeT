from imports import *

# ==== bring in augmentation functions ====
from augmentation import (
    cell_aware_intensity_scale,
    add_gaussian_noise,
    contrast_adjustment,
    histogram_shift,
    gaussian_sharpening,
    gaussian_smoothing,
    zoom_image,
    rotate_image
)



# ========= DATA GENERATOR ==========

class ImageLabelGenerator(Sequence):
    def __init__(self, image_label_pairs, batch_size=2, target_size=(512, 704), augment=False, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.image_label_pairs = image_label_pairs
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_label_pairs))
        self.image_filenames = [img_path for img_path, _ in self.image_label_pairs]  # Added
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.image_label_pairs) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_items = [self.image_label_pairs[i] for i in self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]]
        images, labels = [], []

        for img_path, lbl_path in batch_items:
            try:
                if not os.path.exists(img_path) or not os.path.exists(lbl_path):
                    print(f"[SKIP] Missing file: {img_path} or {lbl_path}")
                    continue

                # ======= IMAGE LOADING =======
                img_ext = os.path.splitext(img_path)[-1].lower()
                if img_ext in ['.tif', '.tiff']:
                    img = tiff.imread(img_path)
                    if img is None or img.size == 0:
                        print(f"[SKIP] Empty image file: {img_path}")
                        continue
                    if img.ndim == 2:
                        img = np.stack([img] * 3, axis=-1)
                    elif img.ndim == 3:
                        if img.shape[0] == 1:
                            img = np.stack([img[0]] * 3, axis=-1)
                        elif img.shape[0] in [2, 3]:
                            img = np.transpose(img, (1, 2, 0))
                        if img.shape[-1] > 3:
                            img = img[:, :, :3]
                else:
                    img = load_img(img_path, target_size=self.target_size, color_mode='rgb')
                    img = img_to_array(img)

                img = img.astype(np.float32)
                img_max = img.max()
                if img.dtype in [np.int8, np.int16, np.int32]:
                    img = (img - img.min()) / (img_max - img.min() + 1e-8)
                elif img.dtype in [np.uint8, np.uint16, np.uint32]:
                    img = img / (img_max if img_max > 1 else 255.0)
                elif img.dtype in [np.float32, np.float64]:
                    img = img / (img_max if img_max > 1 else 1.0)
                img = np.clip(img, 0.0, 1.0)

                img = cv2.resize(img, self.target_size[::-1])

                # ======= LABEL LOADING =======
                raw_lbl = tiff.imread(lbl_path)
                if raw_lbl is None or raw_lbl.size == 0:
                    print(f"[SKIP] Empty label file: {lbl_path}")
                    continue

                if raw_lbl.ndim == 3:
                    if raw_lbl.shape[0] == 1:
                        raw_lbl = raw_lbl[0]
                    elif raw_lbl.shape[-1] == 1:
                        raw_lbl = raw_lbl[..., 0]
                    elif raw_lbl.shape[0] in [2, 3]:
                        raw_lbl = raw_lbl[0]

                lbl = raw_lbl.astype(np.float32)
                lbl_max = lbl.max()
                if raw_lbl.dtype in [np.int8, np.int16, np.int32]:
                    lbl = (lbl - lbl.min()) / (lbl_max - lbl.min() + 1e-8)
                elif raw_lbl.dtype in [np.uint8, np.uint16, np.uint32]:
                    lbl = lbl / (lbl_max if lbl_max > 1 else 255.0)
                elif raw_lbl.dtype in [np.float32, np.float64]:
                    lbl = lbl / (lbl_max if lbl_max > 1 else 1.0)
                lbl = np.clip(lbl, 0.0, 1.0)

                lbl = cv2.resize(lbl, self.target_size[::-1], interpolation=cv2.INTER_NEAREST)
                if lbl.ndim == 2:
                    lbl = np.expand_dims(lbl, axis=-1)

                
                if self.augment:
                    # List of available augmentation functions
                    augmentation_functions = [
                        cell_aware_intensity_scale,
                        add_gaussian_noise,
                        contrast_adjustment,
                        histogram_shift,
                        gaussian_sharpening,
                        gaussian_smoothing,
                        zoom_image,
                        rotate_image
                    ]
                    
                    # Randomly select 2 or 3 augmentations to apply
                    selected_augs = np.random.choice(augmentation_functions, size=np.random.choice([2, 3]), replace=False)
                    
                    for aug in selected_augs:
                        img = aug(img)


                images.append(img)
                labels.append(lbl)

            except Exception as e:
                print(f"[ERROR] Failed to load: {img_path}, {lbl_path} -> {e}")
                continue

        # Fallback in case everything failed
        if len(images) == 0:
            dummy_image = np.zeros((*self.target_size, 3), dtype=np.float32)
            dummy_label = np.zeros((*self.target_size, 1), dtype=np.float32)
            return np.array([dummy_image]), np.array([dummy_label])

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def get_image_label_pairs(image_dir, label_dir):
    valid_exts = ('.tiff', '.tif', '.png', '.jpg', '.jpeg','.bmp')
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]

    image_label_pairs = []
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(image_dir, img_file)

        # Match any label file with the same base name and valid extension
        for ext in valid_exts:
            label_candidate = os.path.join(label_dir, base_name + ext)
            if os.path.exists(label_candidate):
                image_label_pairs.append((img_path, label_candidate))
                break
    return image_label_pairs