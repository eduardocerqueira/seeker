#date: 2024-05-16T17:02:31Z
#url: https://api.github.com/gists/2adf8f4167ce438894c1f6ecb474f3d4
#owner: https://api.github.com/users/Timoha-3000

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Dense, Input, ELU
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam

# Set the environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Функция для генерации данных
def generate_data(noise_level=0.0):
    P = np.zeros((100, 21))
    T = np.zeros((100, 3))
    x = np.linspace(0, 1, 21)
    for i in range(100):
        c = 0.9 * np.random.rand() + 0.1
        a = 0.9 * np.random.rand() + 0.1
        s = 0.9 * np.random.rand() + 0.1
        T[i, :] = [c, a, s]
        P[i, :] = c * np.exp(-((x - a) ** 2 / s))
    if noise_level > 0:
        P += np.random.normal(0, noise_level, P.shape)
    return P, T

def generate_random_data():
    P = np.zeros((100, 21))
    T = np.zeros((100, 3))
    x = np.linspace(0, 1, 21)
    for i in range(100):
        c = 0.9 * np.random.rand() + 0.1
        a = 0.9 * np.random.rand() + 0.1
        s = 0.9 * np.random.rand() + 0.1
        T[i, :] = [c, a, s]
        P[i, :] = 0.9 * np.random.rand() + 0.1
    return P, T

# Функция для создания модели с elu активацией
def create_model():
    model = Sequential([
        Input(shape=(21,)),
        Dense(21, kernel_initializer=HeNormal()),  # Удалите активацию из Dense слоя
        ELU(alpha=1.0),  # Добавьте ELU как отдельный слой
        Dense(15, kernel_initializer=HeNormal()),
        ELU(alpha=1.0),  # Повторите для каждого слоя, где нужна активация ELU
        Dense(3, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


# Функция для отрисовки графика обучения
def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# Функция для визуализации результатов тестирования
def plot_test_results(model, P_test, T_test):
    predictions = model.predict(P_test)
    labels = ['C', 'A', 'S']
    plt.figure(figsize=(14, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.scatter(T_test[:, i], predictions[:, i], alpha=0.6)
        plt.plot([T_test[:, i].min(), T_test[:, i].max()], [T_test[:, i].min(), T_test[:, i].max()], 'k--', lw=2)
        plt.title(f'Testing for parameter {labels[i]}')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.grid(True)
    plt.tight_layout()
    plt.show()


# Основной скрипт
if __name__ == "__main__":
    #P, T = generate_data(noise_level=0.05)  # с небольшим шумом
    #P, T = generate_data()  # без шума
    P, T = generate_random_data()  # случайные данные
    model = create_model()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=80, restore_best_weights=True)
    history = model.fit(P, T, validation_split=0.2, epochs=6000, verbose=0, callbacks=[early_stopping])
    #history = model.fit(P, T, validation_split=0.2, epochs=1000, verbose=0)

    plot_training_history(history)

    P_test, T_test = generate_data()  # Генерация тестовых данных без шума
    plot_test_results(model, P_test, T_test)
    model.save('ssss.h5')
    # Пример предсказания
    x = np.linspace(0, 1, 21)
    c, a, s = 0.2, 0.8, 0.7
    p = c * np.exp(-((x - a) ** 2 / s))
    p = p.reshape(1, -1)
    y_pred = model.predict(p)
    print("Получим:\nY =")
    print(f" {y_pred[0, 0]:.4f} (C)")
    print(f" {y_pred[0, 1]:.4f} (A)")
    print(f" {y_pred[0, 2]:.4f} (S)")
    model.save('spss.h5')