#date: 2025-01-01T16:37:45Z
#url: https://api.github.com/gists/a0b8de83673e38e59ce07c59b25a37ef
#owner: https://api.github.com/users/PieroPaialungaAI

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Build the FFNN model
model = Sequential()
model.add(Dense(10, input_shape=(window_size,), activation='relu'))
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mse')

ind_order = np.random.choice(len(X),size=len(X))
X = X[ind_order]
y = y[ind_order]

X_train, y_train, X_test, y_test = X[:int(0.8*len(X))], y[:int(0.8*len(y))], X[int(0.8*len(X)):], y[int(0.8*len(y)):]


# Train the model
model.fit(X_train, y_train, epochs=100, verbose=1)