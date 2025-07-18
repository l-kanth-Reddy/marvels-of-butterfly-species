import os
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.applications.vgg16 import VGG16# type: ignore
from tensorflow.keras.layers import Dense, Flatten, Dropout# type: ignore
from tensorflow.keras.models import Model# type: ignore
from tensorflow.keras.optimizers import Adam# type: ignore

# Paths to dataset
train_dir = "butterflies/Butterfly Identification/dataset/train"
val_dir = "butterflies/Butterfly Identification/dataset/val"

# Image Data Generator
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow from directory
train_gen = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), class_mode='categorical', batch_size=32)
val_gen = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), class_mode='categorical', batch_size=32)

# Load pre-trained VGG16 model (without top layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False  # Freeze layers

# Custom layers on top
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(len(train_gen.class_indices), activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the modelmodel.fit(train_gen, validation_data=val_gen, epochs=2)
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=1,
    steps_per_epoch=40,       # Lowered
    validation_steps=20       # Lowered
)


# Save model
#model.save("vgg16_model.h5")
model.save("vgg16_model_small2.h5", include_optimizer=False)

print("✅ Model trained and saved as vgg16_model_small2.h5")
