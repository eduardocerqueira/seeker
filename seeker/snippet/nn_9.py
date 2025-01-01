#date: 2025-01-01T16:53:20Z
#url: https://api.github.com/gists/bfef1204f632e3769e4aca8b08492a5f
#owner: https://api.github.com/users/PieroPaialungaAI

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate the data: a second-degree polynomial + sine wave
time_steps = np.linspace(0, 100, 1000)
data = 0.01 * time_steps**2 + np.sin(time_steps)

# Plot the generated data
plt.figure(figsize=(8, 6))
plt.plot(time_steps, data, label='Generated Data')
plt.title("Generated Data")
plt.show()

# Prepare the data for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.reshape(-1, 1))

# Define the train-test split index
train_size = int(len(scaled_data) * 0.8)  # 80% for training, 20% for testing

# Split the data sequentially
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 50:]  # Start slightly earlier to provide enough input for the model

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

sequence_length = 50
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Reshape X for LSTM input (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1)

# Predict on the test set
test_predicted = model.predict(X_test)

# Inverse transform the predictions and original y for comparison
test_predicted = scaler.inverse_transform(test_predicted)

y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Adjust test_time_steps to match y_test_actual length
test_time_steps = time_steps[train_size:train_size + len(y_test_actual)]
train_time_steps = time_steps[:train_size]
# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(train_time_steps, data[:train_size], label='Train Set')
plt.plot(test_time_steps, y_test_actual, label='True - Test Set')
plt.plot(test_time_steps, test_predicted, label='Predicted - Test Set', linestyle='--')
plt.legend()
plt.title("True vs Predicted Values on Test Set")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
