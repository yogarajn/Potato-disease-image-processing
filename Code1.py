import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_PATH = r"C:\Users\ASUS VivoBook 15\Desktop\New folder"
SAVE_PATH = r'C:\Users\ASUS VivoBook 15\Desktop\MyModelFolder'

# Create the directory if it does not exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Load dataset
train_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)
validation_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

# Get class names
class_names = train_dataset.class_names

# Data Augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# Normalize the pixel values
normalization_layer = layers.Rescaling(1./255)

# Augment and normalize datasets
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(data_augmentation(x)), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

# Use a pre-trained model (VGG16) and fine-tune it
base_model = VGG16(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                   include_top=False,
                   weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks for early stopping and reducing learning rate
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,
    callbacks=[early_stopping, reduce_lr]
)

# Save the trained model
model.save(os.path.join(SAVE_PATH, 'my_model.keras'))

# Unfreeze the base model for fine-tuning
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fine-tune the model
fine_tune_epochs = 10
total_epochs = 20 + fine_tune_epochs

history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    callbacks=[early_stopping, reduce_lr]
)

# Save the fine-tuned model
model.save(os.path.join(SAVE_PATH, 'my_fine_tuned_model.keras'))
