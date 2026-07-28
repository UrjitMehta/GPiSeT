from imports import *

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

def get_logger(log_path):
    log_file = open(log_path, "w")
    def logprint(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()
    return logprint, log_file


def safe_get(history, key):
    if key in history.history:
        return history.history[key]
    fallback_len = len(history.history.get('loss', []))
    return [np.nan] * fallback_len

def pad_histories(histories):
    max_len = max(len(h) for h in histories)
    return np.array([np.pad(h, (0, max_len - len(h)), constant_values=np.nan) for h in histories])


# =============================
# Cross-Validation Training
# =============================
image_label_pairs = get_image_label_pairs(train_img_dir, train_lbl_dir)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
total_epochs = 200

# Storage for metrics
all_val_losses, all_val_dice_scores = [], []
all_val_f1_scores, all_val_ap_scores, all_val_afnr_scores = [], [], []

all_train_loss, all_val_loss = [], []
all_train_dice, all_val_dice = [], []
all_train_f1, all_val_f1 = [], []
all_train_ap, all_val_ap = [], []
all_train_afnr, all_val_afnr = [], []

# COCO-related metrics
all_coco_bbox_ap, all_coco_bbox_ap50, all_coco_bbox_ap75 = [], [], []
all_coco_segm_ap, all_coco_segm_ap50, all_coco_segm_ap75 = [], [], []

# Logger
train_log_path = os.path.join(summary_dir, "training_logs.txt")
train_logprint, train_log_file = get_logger(train_log_path)

guidance_cnn = build_guidance_cnn(input_shape=(512, 704, 3))


# =============================
# Cross-validation loop
# =============================
for fold, (train_idx, val_idx) in enumerate(kf.split(image_label_pairs)):
    train_logprint(f"\n========== Fold {fold+1} ==========")

    fold_output_dir = os.path.join(output_dir, f"fold_{fold+1}")
    os.makedirs(fold_output_dir, exist_ok=True)

    summary_path = os.path.join(model_dir, f'fold_{fold+1}_model_summary.txt')
    checkpoint_path = os.path.join(model_dir, f'swinunet_model_kfold_{fold+1}.keras')
    fold_plot_path = os.path.join(loss_dir, f'training_curves_fold{fold+1}.png')

    # Data generators
    train_pairs = [image_label_pairs[i] for i in train_idx]
    val_pairs = [image_label_pairs[i] for i in val_idx]

    train_generator = ImageLabelGenerator(
        train_pairs, batch_size=2, target_size=(512, 704), shuffle=True, augment=False, use_guidance=use_input_guidance, guidance_cnn=guidance_cnn
    )
    val_generator = ImageLabelGenerator(
        val_pairs, batch_size=2, target_size=(512, 704), shuffle=False, augment=False, use_guidance=use_input_guidance, guidance_cnn=guidance_cnn
    )

    # Model
    input_channels = 3 + int(use_input_guidance)
    model = swin_unet(input_size=(512, 704, input_channels))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipvalue=1.0),
        loss=bce_dice_loss,
        metrics=['accuracy', dice_coef, F1Score(), ap_metric, AFNR()]
    )

    with open(summary_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # =============================
    # Callbacks
    # =============================
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(log_dir, f"fold_{fold+1}"), histogram_freq=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=175, min_delta=0.001, restore_best_weights=True
    )

    best_model_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1
    )

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6, verbose=1
    )

    epoch_checkpoint_callback = EpochCheckpoint(
        save_path_base=model_dir, fold_number=fold + 1, interval=10
    )
    
    save_preds_callback = SavePredictionsCallback(
        generator=val_generator,
        save_path=fold_output_dir,
        num_sets=4,
        generate_guiding_map=generate_guiding_map  # returning guidance (H, W, 1)
    )

    sanity_check_callback = OutputSanityCheckCallback(
        generator=val_generator, num_batches_to_check=1
    )

    coco_eval_callback = COCOEvalCallback(
        val_generator=val_generator,
        save_dir=os.path.join(fold_output_dir, "coco_eval"),
        threshold=0.1,
        eval_interval=10
    )

    # =============================
    # Training
    # =============================
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=total_epochs,
        callbacks=[
            early_stop,
            best_model_callback,
            reduce_lr_callback,
            save_preds_callback,
            tensorboard_callback,
            sanity_check_callback,
            epoch_checkpoint_callback,
            coco_eval_callback
        ],
        verbose=1
    )

    # =============================
    # Per-fold evaluation
    # =============================
    val_metrics = model.evaluate(val_generator, verbose=0)
    all_val_losses.append(val_metrics[0])
    all_val_dice_scores.append(val_metrics[2])
    all_val_f1_scores.append(val_metrics[3])
    all_val_ap_scores.append(val_metrics[4])
    all_val_afnr_scores.append(val_metrics[5])

    # Save history
    all_train_loss.append(safe_get(history, 'loss'))
    all_val_loss.append(safe_get(history, 'val_loss'))
    all_train_dice.append(safe_get(history, 'dice_coef'))
    all_val_dice.append(safe_get(history, 'val_dice_coef'))
    all_train_f1.append(safe_get(history, 'f1_score'))
    all_val_f1.append(safe_get(history, 'val_f1_score'))
    all_train_ap.append(safe_get(history, 'average_precision'))
    all_val_ap.append(safe_get(history, 'val_average_precision'))
    all_train_afnr.append(safe_get(history, 'average_false_negative_ratio'))
    all_val_afnr.append(safe_get(history, 'val_average_false_negative_ratio'))

    # Collect COCO metrics
    logs = getattr(coco_eval_callback, "last_logs", {}) or {}
    all_coco_bbox_ap.append(logs.get("coco_bbox_ap", np.nan))
    all_coco_bbox_ap50.append(logs.get("coco_bbox_ap50", np.nan))
    all_coco_bbox_ap75.append(logs.get("coco_bbox_ap75", np.nan))
    all_coco_segm_ap.append(logs.get("coco_segm_ap", np.nan))
    all_coco_segm_ap50.append(logs.get("coco_segm_ap50", np.nan))
    all_coco_segm_ap75.append(logs.get("coco_segm_ap75", np.nan))

    # =============================
    # Per-fold plots
    # =============================
    plt.figure(figsize=(18, 12))
    metrics_to_plot = [
        ('loss', 'Loss'),
        ('dice_coef', 'Dice'),
        ('f1_score', 'F1'),
        ('average_precision', 'AP'),
        ('average_false_negative_ratio', 'AFNR')
    ]
    for i, (m, title) in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 3, i)
        train_vals = history.history.get(m, [np.nan])
        val_vals = history.history.get(f'val_{m}', [np.nan])
        plt.plot(train_vals, label=f'Train {title}')
        plt.plot(val_vals, label=f'Val {title}')
        plt.title(f"Fold {fold+1} - {title}")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.legend()

    if hasattr(coco_eval_callback, "epoch_logs") and coco_eval_callback.epoch_logs:
        if "AP_bbox" in coco_eval_callback.epoch_logs and "AP_segm" in coco_eval_callback.epoch_logs:
            plt.subplot(2, 3, 6)
            plt.plot(coco_eval_callback.epoch_logs["AP_bbox"], label="Val BBox AP")
            plt.plot(coco_eval_callback.epoch_logs["AP_segm"], label="Val Segm AP")
            plt.title(f"Fold {fold+1} - COCO APs")
            plt.xlabel("Eval step")
            plt.ylabel("AP")
            plt.legend()

    plt.tight_layout()
    plt.savefig(fold_plot_path)
    plt.close()

    # Cleanup GPU memory
    tf.keras.backend.clear_session()
    del model
    gc.collect()
    
# =============================
# Cross-validation summary
# =============================
mean_loss, std_loss = np.nanmean(all_val_losses), np.nanstd(all_val_losses)
mean_dice, std_dice = np.nanmean(all_val_dice_scores), np.nanstd(all_val_dice_scores)
mean_f1, std_f1 = np.nanmean(all_val_f1_scores), np.nanstd(all_val_f1_scores)
mean_ap, std_ap = np.nanmean(all_val_ap_scores), np.nanstd(all_val_ap_scores)
mean_afnr, std_afnr = np.nanmean(all_val_afnr_scores), np.nanstd(all_val_afnr_scores)

mean_bbox_ap, std_bbox_ap = np.nanmean(all_coco_bbox_ap), np.nanstd(all_coco_bbox_ap)
mean_bbox_ap50, std_bbox_ap50 = np.nanmean(all_coco_bbox_ap50), np.nanstd(all_coco_bbox_ap50)
mean_bbox_ap75, std_bbox_ap75 = np.nanmean(all_coco_bbox_ap75), np.nanstd(all_coco_bbox_ap75)

mean_segm_ap, std_segm_ap = np.nanmean(all_coco_segm_ap), np.nanstd(all_coco_segm_ap)
mean_segm_ap50, std_segm_ap50 = np.nanmean(all_coco_segm_ap50), np.nanstd(all_coco_segm_ap50)
mean_segm_ap75, std_segm_ap75 = np.nanmean(all_coco_segm_ap75), np.nanstd(all_coco_segm_ap75)

train_logprint("\n=== Cross-Validation Summary ===")
train_logprint(f"Avg Loss: {mean_loss:.4f} +/- {std_loss:.4f}")
train_logprint(f"Avg Dice Coef: {mean_dice:.4f} +/- {std_dice:.4f}")
train_logprint(f"Avg F1 Score: {mean_f1:.4f} +/- {std_f1:.4f}")
train_logprint(f"Avg Average Precision: {mean_ap:.4f} +/- {std_ap:.4f}")
train_logprint(f"Avg AFNR: {mean_afnr:.4f} +/- {std_afnr:.4f}")
train_logprint(f"Avg COCO BBox AP: {mean_bbox_ap:.4f} +/- {std_bbox_ap:.4f}")
train_logprint(f"Avg COCO BBox AP50: {mean_bbox_ap50:.4f} +/- {std_bbox_ap50:.4f}")
train_logprint(f"Avg COCO BBox AP75: {mean_bbox_ap75:.4f} +/- {std_bbox_ap75:.4f}")
train_logprint(f"Avg COCO Segm AP: {mean_segm_ap:.4f} +/- {std_segm_ap:.4f}")
train_logprint(f"Avg COCO Segm AP50: {mean_segm_ap50:.4f} +/- {std_segm_ap50:.4f}")
train_logprint(f"Avg COCO Segm AP75: {mean_segm_ap75:.4f} +/- {std_segm_ap75:.4f}")
train_log_file.close()

# =============================
# Combined curves
# =============================
all_train_loss_arr  = pad_histories(all_train_loss)
all_val_loss_arr    = pad_histories(all_val_loss)
all_train_dice_arr  = pad_histories(all_train_dice)
all_val_dice_arr    = pad_histories(all_val_dice)
all_train_f1_arr    = pad_histories(all_train_f1)
all_val_f1_arr      = pad_histories(all_val_f1)
all_train_ap_arr    = pad_histories(all_train_ap)
all_val_ap_arr      = pad_histories(all_val_ap)
all_train_afnr_arr  = pad_histories(all_train_afnr)
all_val_afnr_arr    = pad_histories(all_val_afnr)

x = np.arange(1, all_train_loss_arr.shape[1] + 1)
plt.figure(figsize=(18, 12))
metrics_to_plot = [
    (all_train_loss_arr, all_val_loss_arr, 'Loss'),
    (all_train_dice_arr, all_val_dice_arr, 'Dice'),
    (all_train_f1_arr, all_val_f1_arr, 'F1'),
    (all_train_ap_arr, all_val_ap_arr, 'AP'),
    (all_train_afnr_arr, all_val_afnr_arr, 'AFNR')
]

def plot_with_std(x, data, label, color):
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    plt.plot(x, mean, label=label, color=color)
    plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

for i, (train_data, val_data, title) in enumerate(metrics_to_plot, 1):
    plt.subplot(2, 3, i)
    plot_with_std(x, train_data, f'Train {title}', 'blue')
    plot_with_std(x, val_data, f'Val {title}', 'orange')
    plt.title(f"Combined {title}")
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.legend()

plt.tight_layout()
combined_plot_path = os.path.join(loss_dir, 'combined_training_curves.png')
plt.savefig(combined_plot_path)
plt.close()
