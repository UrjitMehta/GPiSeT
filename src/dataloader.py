from imports import *

# ========= DATA GENERATOR ==========

class ImageLabelGenerator(Sequence):
    def __init__(self, image_label_pairs, batch_size=2, target_size=(512, 704), augment=False, shuffle=True, use_guidance=False, guidance_cnn=None):
        self.image_label_pairs = image_label_pairs
        self.image_filenames = [img for img, _ in image_label_pairs]
        self.batch_size = batch_size
        self.target_size = target_size
        self.augment = augment
        self.shuffle = shuffle
        self.use_guidance = use_guidance
        self.guidance_cnn = guidance_cnn
        self.indexes = np.arange(len(self.image_label_pairs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.image_label_pairs) / self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # ---- Augmentation handler ----
    def apply_augmentations(self, img, lbl):
        # geometric transforms (apply to both)
        if np.random.rand() < 0.5:
            k = np.random.choice([0, 1, 2, 3])
            img = rotate_image(img, k)
            lbl = rotate_image(lbl, k)

        if np.random.rand() < 0.5:
            img = zoom_image(img)
            lbl = zoom_image(lbl)

        # color/intensity transforms (image only)
        if np.random.rand() < 0.5:
            img = cell_aware_intensity_scale(img)
        if np.random.rand() < 0.5:
            img = add_gaussian_noise(img)
        if np.random.rand() < 0.5:
            img = contrast_adjustment(img)
        if np.random.rand() < 0.3:
            img = gaussian_smoothing(img)

        return img, lbl

    # ---- Safe label loader ----
    def load_label_safe(self, lbl_path):
        lbl = tiff.imread(lbl_path)
        if lbl is None or lbl.size == 0:
            raise ValueError(f"Empty label file: {lbl_path}")

        # Normalize shape (handle (H,W), (1,H,W), (H,W,1), (C,H,W))
        if lbl.ndim == 3:
            if lbl.shape[0] == 1:
                lbl = lbl[0]
            elif lbl.shape[-1] == 1:
                lbl = lbl[..., 0]
            elif lbl.shape[0] in [2, 3]:
                lbl = lbl[0]

        # Ensure single channel
        if lbl.ndim == 2:
            lbl = np.expand_dims(lbl, -1)

        lbl = lbl.astype(np.float32)
        lbl /= (lbl.max() + 1e-8)
        lbl = np.clip(lbl, 0, 1)
        lbl = cv2.resize(lbl, self.target_size[::-1], interpolation=cv2.INTER_NEAREST)

        # Ensure correct shape again
        if lbl.ndim == 2:
            lbl = np.expand_dims(lbl, -1)

        return lbl

    # ---- Main generator ----
    def __getitem__(self, idx):
        batch_items = [
            self.image_label_pairs[i] 
            for i in self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        ]

        images, labels = [], []

        for img_path, lbl_path in batch_items:
            try:
                # --- Load image ---
                if not os.path.exists(img_path) or not os.path.exists(lbl_path):
                    print(f"[SKIP] Missing file: {img_path} or {lbl_path}")
                    continue

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
                img /= (img.max() + 1e-8)
                img = np.clip(img, 0, 1)
                img = cv2.resize(img, self.target_size[::-1])

                # --- Load label safely ---
                lbl = self.load_label_safe(lbl_path)

                # --- Guidance channel ---
                if self.use_guidance:
                    if self.guidance_cnn is not None:
                        guidance_map = self.guidance_cnn.predict(
                            np.expand_dims(img, 0), batch_size=1, verbose=0
                        )[0]
                    else:
                        guidance_map = generate_guiding_map(img)
                    img = np.concatenate([img, guidance_map], axis=-1)

                # --- Apply augmentations ---
                if self.augment:
                    img, lbl = self.apply_augmentations(img, lbl)
                    # enforce consistent shape
                    img = cv2.resize(img, self.target_size[::-1], interpolation=cv2.INTER_LINEAR)
                    lbl = cv2.resize(lbl, self.target_size[::-1], interpolation=cv2.INTER_NEAREST)
                    if lbl.ndim == 2:
                        lbl = np.expand_dims(lbl, -1)

                # --- Append to batch ---
                images.append(img)
                labels.append(lbl)

            except Exception as e:
                print(f"[ERROR] Failed to load {img_path} -> {e}")
                continue

        # --- Fallback if empty ---
        if len(images) == 0:
            dummy_image = np.zeros((*self.target_size, 3 + int(self.use_guidance)), dtype=np.float32)
            dummy_label = np.zeros((*self.target_size, 1), dtype=np.float32)
            return np.array([dummy_image]), np.array([dummy_label])

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)
           
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
