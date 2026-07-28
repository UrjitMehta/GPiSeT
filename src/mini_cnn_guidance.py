from imports import *

# ===============================
# Mini-CNN for Guidance
# ===============================
def build_guidance_cnn(input_shape=(512, 704, 3)):
    inputs = layers.Input(shape=input_shape)
    # Encoder
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    # Decoder
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.UpSampling2D(2)(x)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
    return models.Model(inputs, outputs, name="guidance_cnn")


# ===============================
# Input-level guidance map
# ===============================
def generate_guiding_map(img):
    if img.ndim == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
    elif img.ndim == 3:
        img_gray = img.max(axis=-1)
    else:
        img_gray = img.astype(np.float32)

    img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)
    img_contrast = exposure.equalize_adapthist(img_gray, clip_limit=0.03)
    img_blur = cv2.GaussianBlur(img_contrast, (3, 3), 0)

    try:
        t = threshold_otsu(img_blur)
    except ValueError:
        t = 0.5
    mask = np.maximum(img_blur - t, 0)
    mask = opening(mask, footprint_rectangle((3, 3)))
    mask = closing(mask, footprint_rectangle((3, 3)))
    mask_norm = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask_norm[..., np.newaxis]


# ===============================
# Pre-training Guidance CNN
# ===============================
def pretrain_guidance_cnn(cnn_model, image_paths, target_size=(512, 704), batch_size=4, epochs=5):
    imgs, heuristics = [], []
    for img_path in image_paths:
        img = load_img(img_path, target_size=target_size)
        img = img_to_array(img).astype(np.float32) / 255.0
        heuristic_map = generate_guiding_map(img)
        imgs.append(img)
        heuristics.append(heuristic_map)

    X = np.array(imgs, dtype=np.float32)
    y = np.array(heuristics, dtype=np.float32)

    cnn_model.compile(optimizer=Adam(1e-3), loss=MeanSquaredError())
    cnn_model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=1)
    return cnn_model
