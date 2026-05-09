import os
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.config.optimizer.set_experimental_options({'layout_optimizer': False, 'disable_meta_optimizer': False})
tf.config.optimizer.set_jit(False)


import glob
import random
import subprocess
from pathlib import Path
import gc

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

# ---------------------------------------------------------
# Enable GPU Memory Growth safely
# ---------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPUs.")
    except RuntimeError as e:
        print(e)

# ---------------------------------------------------------
# HPC Paths Configuration
# ---------------------------------------------------------
HPC_ROOT = Path('/scratch/lustre/home/maza9905/kursinis')

zip_path = HPC_ROOT / 'exports' / 'mel_stitched_224x448.7z'
extract_path = HPC_ROOT / 'extracted_data'
csv_file_path = HPC_ROOT / 'artifacts' / 'combined_patient_data.csv'

SPLIT_ROOT = HPC_ROOT / 'artifacts' / 'splits'
MODEL_OUTPUT_ROOT = HPC_ROOT / 'artifacts' / 'model_outputs'
MODEL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
extract_path.mkdir(parents=True, exist_ok=True)

LOCAL_DATA_ROOT = extract_path / 'mel_stitched_224x448'

conda_bin_7z = os.path.join(os.environ.get('CONDA_PREFIX', '/usr/bin'), 'bin', '7z')

# ---------------------------------------------------------
# Data Extraction
# ---------------------------------------------------------
if not LOCAL_DATA_ROOT.exists():
    print(f"Extracting data to {extract_path}...")
    subprocess.run([conda_bin_7z, 'x', str(zip_path), f'-o{extract_path}', '-y'], check=True)
else:
    print(f"Data already extracted at {LOCAL_DATA_ROOT}")

# ---------------------------------------------------------
# Augmentation Functions
# ---------------------------------------------------------
def adjust_brightness_contrast(image):
    alpha = random.uniform(0.5, 1.5)
    beta = random.uniform(-50, 50)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def frequency_mask(image):
    img = image.copy()
    h, w = img.shape[:2]
    mask_width = int(w * random.uniform(0, 0.5))
    x_start = random.randint(0, max(0, w - mask_width))
    img[:, x_start:x_start + mask_width] = 0
    return img

def gaussian_blur(image):
    k = random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (k, k), 0)

def time_shift(image):
    img = image.copy()
    h, w = img.shape[:2]
    shift = int(w * random.uniform(-0.2, 0.2))
    shifted = np.zeros_like(img)
    if shift > 0:
        shifted[:, shift:] = img[:, :w - shift]
    elif shift < 0:
        shifted[:, :w + shift] = img[:, -shift:]
    else:
        shifted = img
    return shifted

def cutout(image):
    img = image.copy()
    h, w = img.shape[:2]
    num_holes = random.randint(1, 8)
    for _ in range(num_holes):
        hole_w = random.randint(10, 45)
        hole_h = random.randint(10, 45)
        x = random.randint(0, max(0, w - hole_w))
        y = random.randint(0, max(0, h - hole_h))
        img[y:y + hole_h, x:x + hole_w] = 0
    return img

def apply_augmentations(image_np: np.ndarray) -> np.ndarray:
    img = image_np.astype(np.uint8)
    if random.random() < 0.8: img = adjust_brightness_contrast(img)
    if random.random() < 0.8: img = frequency_mask(img)
    if random.random() < 0.6: img = gaussian_blur(img)
    if random.random() < 0.6: img = time_shift(img)
    if random.random() < 0.6: img = cutout(img)
    return img.astype(np.float32)

def augment_tf(image: tf.Tensor, label: tf.Tensor):
    def _augment(img_tensor):
        return tf.numpy_function(func=apply_augmentations, inp=[img_tensor], Tout=tf.float32)
    image = _augment(image)
    image.set_shape([IMG_HEIGHT, IMG_WIDTH, 3]) 
    return image, label

# ---------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------
SEED = 1234
IMG_HEIGHT = 224
IMG_WIDTH = 448
NUM_CLASSES = 2
PER_REPLICA_BATCH_SIZE = 32
EPOCHS = 10
TRAINING_MODALITY = 'grid'

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Initialize Multi-GPU Strategy
strategy = tf.distribute.MirroredStrategy()
print(f"Number of GPUs in sync: {strategy.num_replicas_in_sync}")

# Scale Batch Size by number of GPUs
GLOBAL_BATCH_SIZE = PER_REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync

# Load CSV and Images
df_all_stitched = pd.read_csv(csv_file_path)
all_images = glob.glob(str(LOCAL_DATA_ROOT / '**/*.png'), recursive=True)

paths_df = pd.DataFrame({'absolute_path': all_images})
paths_df['Patient'] = paths_df['absolute_path'].apply(lambda x: int(Path(x).parent.name))
print(f"Found {len(paths_df)} images across {paths_df['Patient'].nunique()} patients")

def load_grid_image(file_path, label):
    image_bytes = tf.io.read_file(file_path)
    image = tf.image.decode_png(image_bytes, channels=3)
    image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
    return image, label

# ---------------------------------------------------------
# Cross-Validation Loop
# ---------------------------------------------------------
patient_meta = df_all_stitched[['Patient', 'Outcome']].drop_duplicates()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

all_fold_results =[]
oof_predictions_list = []

for fold, (train_idx, test_idx) in enumerate(skf.split(patient_meta, patient_meta['Outcome'])):
    print(f"\n{'='*20} STARTING FOLD {fold+1} {'='*20}")

    tf.keras.backend.clear_session()
    gc.collect()

    train_patients_full = patient_meta.iloc[train_idx]
    test_patients = patient_meta.iloc[test_idx]

    train_patients, val_patients = train_test_split(
        train_patients_full, test_size=0.2, stratify=train_patients_full['Outcome'], random_state=SEED
    )

    train_split_df = df_all_stitched[df_all_stitched['Patient'].isin(train_patients['Patient'])].copy()
    val_split_df = df_all_stitched[df_all_stitched['Patient'].isin(val_patients['Patient'])].copy()
    test_split_df = df_all_stitched[df_all_stitched['Patient'].isin(test_patients['Patient'])].copy()

    label_map = {'Good': 0, 'Poor': 1}
    for df_split in[train_split_df, val_split_df, test_split_df]:
        df_split['label'] = df_split['Outcome'].map(label_map)

    train_labels = train_split_df.merge(paths_df, on='Patient')['label'].to_numpy()
    unique = np.unique(train_labels)
    if len(unique) > 1:
        class_weights = compute_class_weight(class_weight='balanced', classes=unique, y=train_labels)
        class_weight_dict = {int(c): float(w) for c, w in zip(unique, class_weights)}
    else:
        class_weight_dict = {int(unique[0]): 1.0}

    fold_suffix = f"fold_{fold+1}"
    checkpoint_path = MODEL_OUTPUT_ROOT / f'best_model_{TRAINING_MODALITY}_{fold_suffix}.keras'
    history_path = MODEL_OUTPUT_ROOT / f'history_{TRAINING_MODALITY}_{fold_suffix}.csv'

    def get_ds(df, shuffle_data=False):
        merged_df = df.merge(paths_df, on='Patient')
        paths = merged_df['absolute_path'].to_numpy()
        labels = merged_df['label'].to_numpy()
        
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle_data:
            ds = ds.shuffle(len(paths), seed=SEED)
            
        ds = ds.map(load_grid_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        if shuffle_data:
            ds = ds.map(augment_tf, num_parallel_calls=tf.data.AUTOTUNE)
            
        return ds.batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    ds_train = get_ds(train_split_df, shuffle_data=True)
    ds_val = get_ds(val_split_df)
    ds_test = get_ds(test_split_df)

    # Multi-GPU scope setup
    with strategy.scope():
        base_model = EfficientNetV2S(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        x = layers.GlobalAveragePooling2D()(base_model.output)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
        model = tf.keras.Model(base_model.input, outputs)

        model.compile(
            optimizer='adam', 
            loss='binary_crossentropy', 
            metrics=['accuracy', tf.keras.metrics.SensitivityAtSpecificity(0.95, name='tpr_at_fpr05')]
        )

    callbacks = [
        ModelCheckpoint(filepath=str(checkpoint_path), monitor='val_tpr_at_fpr05', mode='max', save_best_only=True),
        EarlyStopping(monitor='val_tpr_at_fpr05', mode='max', patience=3, restore_best_weights=True),
        CSVLogger(filename=str(history_path))
    ]

    model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, callbacks=callbacks, class_weight=class_weight_dict, verbose=1)

    best_model = tf.keras.models.load_model(str(checkpoint_path))
    preds = best_model.predict(ds_test).flatten()

    res_df = test_split_df.merge(paths_df, on='Patient').copy()
    res_df['prob_poor'] = preds
    patient_res = res_df.groupby(['Patient', 'label'])['prob_poor'].mean().reset_index()

    # Track which fold this patient was evaluated in
    patient_res['fold'] = fold + 1
    oof_predictions_list.append(patient_res)

    auc = roc_auc_score(patient_res['label'], patient_res['prob_poor'])
    all_fold_results.append({'fold': fold+1, 'auc': auc})
    print(f"Fold {fold+1} Patient-Level AUC: {auc:.4f}")

    del model, base_model, best_model, ds_train, ds_val, ds_test
    gc.collect()

cv_results_df = pd.DataFrame(all_fold_results)
print(cv_results_df)
print(f"Mean CV AUC: {cv_results_df['auc'].mean():.4f}")

# Combine all 5 folds of test predictions into a single DataFrame
oof_df = pd.concat(oof_predictions_list, ignore_index=True)

oof_save_path = MODEL_OUTPUT_ROOT / f'oof_predictions_cnn1_{TRAINING_MODALITY}.csv'
oof_df.to_csv(oof_save_path, index=False)

print(f"\nSUCCESS: OOF predictions saved for {len(oof_df)} patients to {oof_save_path}")