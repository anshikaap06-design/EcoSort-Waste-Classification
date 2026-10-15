
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Project settings
DATA_DIR = Path("dataset")
MODEL_DIR = Path("model")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 5
SEED = 42

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Verify the dataset
if not DATA_DIR.exists():
    raise FileNotFoundError("dataset folder not found!")

# Load training images: 80%
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Load validation images: 20%
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Detected classes:", class_names)

# Save labels for later predictions
with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(class_names, f)

# Load a pretrained image model
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Augment training images
augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1)
])

# Build classification model
inputs = layers.Input(shape=(224, 224, 3))
x = augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# Save the trained model
model.save(MODEL_DIR / "waste_model.keras")

print("Training completed successfully!")
print("Model saved at:", MODEL_DIR / "waste_model.keras")