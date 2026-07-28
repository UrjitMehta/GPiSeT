from imports import *

# ========= DATA GENERATOR ==========

class ImageLabelGenerator(Sequence):
       def __init__(self, image_label_pairs, batch_size=2, target_size=(512, 704), augment=False, shuffle=True, use_guidance=False, **kwargs):
           super().__init__(**kwargs)
           self.image_label_pairs = image_label_pairs
           self.batch_size = batch_size
           self.target_size = target_size
           self.augment = augment
           self.shuffle = shuffle
           self.use_guidance = use_guidance
           self.indexes = np.arange(len(self.image_label_pairs))
           self.image_filenames = [img_path for img_path, _ in self.image_label_pairs]
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
       
                   # --- IMAGE LOADING ---
                   img_ext = os.path.splitext(img_path)[-1].lower()
                   img = None
                   if img_ext in ['.tif', '.tiff']:
                       img = tiff.imread(img_path)
                       # some broken tiffs may return None or empty
                       if img is None or img.size == 0:
                           print(f"[ERROR] Empty TIFF: {img_path}")
                           continue
                       if img.ndim == 2:  # grayscale
                           img = np.stack([img] * 3, axis=-1)
                       elif img.ndim == 3:
                           if img.shape[0] in [1, 2, 3]:  # channels first
                               img = np.transpose(img, (1, 2, 0))
                           if img.shape[-1] > 3:
                               img = img[:, :, :3]
                   else:
                       img = load_img(img_path, target_size=self.target_size, color_mode='rgb')
                       img = img_to_array(img)
       
                   # --- SAFETY CHECK BEFORE RESIZE ---
                   if img is None or img.size == 0:
                       print(f"[ERROR] Invalid image data at {img_path}")
                       continue
       
                   img = img.astype(np.float32) / 255.0
                   img = cv2.resize(img, self.target_size[::-1], interpolation=cv2.INTER_LINEAR)
       
                   # --- INPUT GUIDANCE ---
                   if self.use_guidance:
                       guiding_map = generate_guiding_map(img)
                       img = np.concatenate([img, guiding_map], axis=-1)  # shape (H,W,4)
       
                   # --- LABEL LOADING ---
                   lbl = tiff.imread(lbl_path).astype(np.float32)
                   if lbl is None or lbl.size == 0:
                       print(f"[ERROR] Empty label TIFF: {lbl_path}")
                       continue
       
                   if lbl.ndim == 3:
                       if lbl.shape[0] in [1, 2, 3]:
                           lbl = lbl[0]
                       elif lbl.shape[-1] == 1:
                           lbl = lbl[..., 0]
                   lbl = lbl / (lbl.max() if lbl.max() > 1 else 1.0)
                   lbl = cv2.resize(lbl, self.target_size[::-1], interpolation=cv2.INTER_NEAREST)
                   if lbl.ndim == 2:
                       lbl = np.expand_dims(lbl, axis=-1)
       
                   images.append(img)
                   labels.append(lbl)
       
               except Exception as e:
                   print(f"[ERROR] Failed to load: {img_path}, {lbl_path} -> {e}")
                   continue
       
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
