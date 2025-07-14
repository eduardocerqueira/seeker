#date: 2025-07-14T17:00:10Z
#url: https://api.github.com/gists/1638152e4617212092543806e9662709
#owner: https://api.github.com/users/aniya-mitchell

# CIFAR-10 CNN Classifier with Data Augmentation and Batch Normalization

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)
y_test = to_categorical(y_test, 10)

# Base model
model = Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
    Activation('relu'),
    Conv2D(32, (3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), padding='same'),
    Activation('relu'),
    Conv2D(64, (3, 3), padding='same'),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512),
    Activation('relu'),
    Dropout(0.5),
    Dense(10),
    Activation('softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Save model in new Keras format
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_val, y_val), callbacks=[checkpoint])

# Plot loss for base model
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss (Base Model)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Evaluate base model
print("Base Model Evaluation")
print("Train:", model.evaluate(x_train, y_train))
print("Validation:", model.evaluate(x_val, y_val))

# Data Augmentation
model_aug = clone_model(model)
model_aug.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

checkpoint_aug = ModelCheckpoint('best_model_aug.keras', save_best_only=True, monitor='val_loss', mode='min')
history_aug = model_aug.fit(datagen.flow(x_train, y_train, batch_size=32),
                            epochs=50,
                            validation_data=(x_val, y_val),
                            callbacks=[checkpoint_aug])

# Plot loss for augmented model
plt.plot(history_aug.history['loss'], label='Train Loss')
plt.plot(history_aug.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss (Augmented)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Evaluate augmented model
print("Augmented Model Evaluation")
print("Train:", model_aug.evaluate(x_train, y_train))
print("Validation:", model_aug.evaluate(x_val, y_val))

# Model with Batch Normalization
def build_model_with_bn():
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', use_bias=False, input_shape=(32, 32, 3)),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, (3, 3), padding='same', use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), padding='same', use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same', use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(512, use_bias=False),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),
        Dense(10),
        Activation('softmax')
    ])
    return model

model_bn = build_model_with_bn()
model_bn.compile(optimizer=Adam(learning_rate=0.01),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

checkpoint_bn = ModelCheckpoint('best_model_bn.keras', save_best_only=True, monitor='val_loss', mode='min')
history_bn = model_bn.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_val, y_val), callbacks=[checkpoint_bn])

# Plot loss for batch norm model
plt.plot(history_bn.history['loss'], label='Train Loss')
plt.plot(history_bn.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss (Batch Norm)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Evaluate batch norm model
print("Batch Norm Model Evaluation")
print("Train:", model_bn.evaluate(x_train, y_train))
print("Validation:", model_bn.evaluate(x_val, y_val))

"""g. In step e, the base model without data augmentation shows a steadily increasing validation loss after a certain point, despite the training loss continuing to decrease, indicateing overfitting. In step f, the model trained with data augmentation maintains a lower and more stable validation loss throughout the training process, suggesting a better generalization and reduced overfitting.

i. in step h, the fuctuating validation loss could suggests that the model might be struggling to generalize, likely due to the higher learning rate used during training. Overall, while the batch normalization model demonstrates better training dynamics, fine-tunin
"""

# Combined Loss Plot for All Models

plt.figure(figsize=(12, 6))

# Base Model
plt.plot(history.history['val_loss'], label='Base Model (Val Loss)', linestyle='--')
plt.plot(history.history['loss'], label='Base Model (Train Loss)', linestyle='-')

# Augmented Model
plt.plot(history_aug.history['val_loss'], label='Augmented (Val Loss)', linestyle='--')
plt.plot(history_aug.history['loss'], label='Augmented (Train Loss)', linestyle='-')

# BatchNorm Model
plt.plot(history_bn.history['val_loss'], label='BatchNorm (Val Loss)', linestyle='--')
plt.plot(history_bn.history['loss'], label='BatchNorm (Train Loss)', linestyle='-')

plt.title('Training and Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()