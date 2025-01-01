#date: 2025-01-01T16:42:27Z
#url: https://api.github.com/gists/75717d09202068e4326d2af8130f1aae
#owner: https://api.github.com/users/PieroPaialungaAI

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense



# Build the 1D CNN model
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(len(time_steps), 1)))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, verbose=1, validation_split=0.2)
