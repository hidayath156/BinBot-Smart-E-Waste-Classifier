import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF warnings

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
import pickle

# =====================================================
# PATHS (MAKE SURE THESE MATCH YOUR DATASET)
# =====================================================
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/valid"

FINAL_MODEL_PATH = "ewaste_efficientnetb0_final.h5"
BEST_MODEL_PATH = "best_efficientnetb0_model.h5"
CLASS_INDEX_PATH = "class_indices.pkl"

# =====================================================
# TRAINING CONFIG
# =====================================================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
FINE_TUNE_EPOCHS = 10

# =====================================================
# SAFETY CHECKS
# =====================================================
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"❌ Train directory not found: {TRAIN_DIR}")

if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(f"❌ Validation directory not found: {VAL_DIR}")

# =====================================================
# DATA GENERATORS
# =====================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    zoom_range=0.25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_gen.num_classes
print("✅ Classes found:", train_gen.class_indices)

# =====================================================
# SAVE CLASS INDICES (CRITICAL FOR PREDICTION)
# =====================================================
with open(CLASS_INDEX_PATH, "wb") as f:
    pickle.dump(train_gen.class_indices, f)

# =====================================================
# BUILD EFFICIENTNETB0 MODEL
# =====================================================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # Phase 1: freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# =====================================================
# CALLBACKS (VERY IMPORTANT)
# =====================================================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# =====================================================
# PHASE 1: TRANSFER LEARNING
# =====================================================
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n🚀 Starting Phase 1: Transfer Learning")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =====================================================
# PHASE 2: FINE-TUNING
# =====================================================
print("\n🔧 Starting Phase 2: Fine-Tuning")

base_model.trainable = True

# Freeze most layers, fine-tune top ones
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks
)

# =====================================================
# SAVE FINAL MODEL (FULL MODEL, NOT JUST WEIGHTS)
# =====================================================
model.save(FINAL_MODEL_PATH)

print("\n✅ Training completed successfully!")
print("📦 Best model saved as :", BEST_MODEL_PATH)
print("📦 Final model saved as:", FINAL_MODEL_PATH)
print("📦 Class indices saved:", CLASS_INDEX_PATH)
