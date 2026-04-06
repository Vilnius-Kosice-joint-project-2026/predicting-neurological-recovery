from pathlib import Path, PurePosixPath
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from google.colab import drive
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from tensorflow.keras import mixed_precision


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


SEED = 1234
IMG_SIZE = 224
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 30
TRAINING_MODALITY = 'grid'

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
mixed_precision.set_global_policy('mixed_float16')

print('Mixed precision policy:', mixed_precision.global_policy())

drive.mount('/content/drive')

# Configure paths in Google Drive
PROJECT_ROOT = Path('/content/drive/MyDrive/predicting-neurological-recovery')
SPLIT_ROOT = PROJECT_ROOT / 'artifacts' / 'splits'

DATA_ROOT = PROJECT_ROOT / 'exports'
path_column = 'grid_relative_path'

MODEL_OUTPUT_ROOT = PROJECT_ROOT / 'artifacts' / 'model_outputs'
MODEL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

print('Split root:', SPLIT_ROOT)
print(f'Data root ({TRAINING_MODALITY}):', DATA_ROOT)
print('Model output root:', MODEL_OUTPUT_ROOT)

# Load prepared split artifacts
train_split_df = pd.read_parquet(SPLIT_ROOT / 'train_split.parquet')
val_split_df = pd.read_parquet(SPLIT_ROOT / 'val_split.parquet')
test_split_df = pd.read_parquet(SPLIT_ROOT / 'test_split.parquet')

for split_name, split_df in [('train', train_split_df), ('val', val_split_df), ('test', test_split_df)]:
    # Replace Windows-style backslashes with POSIX forward slashes
    split_df[path_column] = split_df[path_column].str.replace('\\', '/', regex=False)
    split_df['absolute_path'] = split_df[path_column].map(lambda path_value: str(DATA_ROOT / path_value))
    print(split_name, 'rows=', len(split_df), 'labels=', split_df['label'].value_counts().to_dict())

# Build arrays for tf.data
train_x = train_split_df['absolute_path'].astype(str).to_numpy()
train_y = train_split_df['label'].to_numpy()
train_x, train_y = shuffle(train_x, train_y, random_state=SEED)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_y), y=train_y)
class_weight_dict = dict(enumerate(class_weights))
print('Class weights:', class_weight_dict)

val_x = val_split_df['absolute_path'].astype(str).to_numpy()
val_y = val_split_df['label'].to_numpy()
test_x = test_split_df['absolute_path'].astype(str).to_numpy()
test_y = test_split_df['label'].to_numpy()

print('train:', train_x.shape, train_y.shape)
print('val:', val_x.shape, val_y.shape)
print('test:', test_x.shape, test_y.shape)

# Build tf.data datasets
def load_grid_image(file_path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(file_path)
    image = tf.image.decode_png(image_bytes, channels=3)  # force 3 channels 
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    return image, label

load_func = load_grid_image

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = train_dataset.shuffle(buffer_size=len(train_x), reshuffle_each_iteration=True)
train_dataset = train_dataset.map(load_func, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
validation_dataset = validation_dataset.map(load_func, num_parallel_calls=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_dataset = test_dataset.map(load_func, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Build data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomZoom(height_factor=0.1, width_factor=0.1),
    layers.RandomRotation(factor=0.05),
], name="data_augmentation")

# Build and compile EfficientNetB0 with ImageNet weights
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
augmented = data_augmentation(inputs)
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=augmented)

# FREEZE THE PRE-TRAINED WEIGHTS HERE
base_model.trainable = False

# 2. Add our own pooling and classification layer for 2 classes
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()

# Train with callbacks and save best checkpoint
checkpoint_path = MODEL_OUTPUT_ROOT / f'best_efficientnetb0_{TRAINING_MODALITY}.keras'
latest_weights_path = MODEL_OUTPUT_ROOT / f'latest_efficientnetb0_{TRAINING_MODALITY}.weights.h5'
final_model_path = MODEL_OUTPUT_ROOT / f'final_efficientnetb0_{TRAINING_MODALITY}.keras'
history_path = MODEL_OUTPUT_ROOT / f'history_{TRAINING_MODALITY}.csv'

if latest_weights_path.exists():
    model.load_weights(latest_weights_path)
    print('Resumed from latest weights:', latest_weights_path)
else:
    print('No previous latest weights found. Starting fresh run.')

callbacks = [
    ModelCheckpoint(filepath=str(checkpoint_path), monitor='val_loss', save_best_only=True),
    ModelCheckpoint(filepath=str(latest_weights_path), save_weights_only=True, save_best_only=False),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    CSVLogger(filename=str(history_path), append=True),
]

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    verbose=1,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

model.save(final_model_path)
print('Saved:', checkpoint_path)
print('Saved:', latest_weights_path)
print('Saved:', final_model_path)
print('Saved:', history_path)

# ==========================================
# PHASE 2: FINE-TUNING THE ENTIRE MODEL
# ==========================================
print("\n--- Starting Phase 2: Fine-tuning ---")

base_model.trainable = True

for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

# 2. Recompile with a VERY low learning rate (e.g., 1e-5)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model.summary()

# 3. Train for additional epochs
FINE_TUNE_EPOCHS = 20

# We append to the existing history log so you don't overwrite Phase 1's metrics
history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history.epoch[-1],  # Start from where the previous fit left off
    verbose=1,
    callbacks=callbacks,
    class_weight=class_weight_dict
)

# 4. Save the fine-tuned model
final_finetuned_path = MODEL_OUTPUT_ROOT / f'final_finetuned_efficientnetb0_{TRAINING_MODALITY}.keras'
model.save(final_finetuned_path)
print('Saved fine-tuned model:', final_finetuned_path)


# Evaluate and export predictions
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
probabilities = model.predict(test_dataset, verbose=1)
predicted_label = probabilities.argmax(axis=1)

test_results_df = test_split_df.copy()
test_results_df['predicted_label'] = predicted_label
test_results_df['prob_poor'] = probabilities[:, 0]
test_results_df['prob_good'] = probabilities[:, 1]

metrics_path = MODEL_OUTPUT_ROOT / f'test_metrics_{TRAINING_MODALITY}.csv'
predictions_path = MODEL_OUTPUT_ROOT / f'test_predictions_{TRAINING_MODALITY}.csv'
pd.DataFrame([{'test_loss': test_loss, 'test_accuracy': test_accuracy}]).to_csv(metrics_path, index=False)
test_results_df.to_csv(predictions_path, index=False)

print('Test accuracy:', test_accuracy)
print('Saved:', metrics_path)
print('Saved:', predictions_path)

# 1. Compute the confusion matrix
# Note: test_y are the true labels, predicted_label are the model predictions
cm = confusion_matrix(test_y, predicted_label)

# 2. Normalize the confusion matrix by row (true labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

print(cm_normalized)