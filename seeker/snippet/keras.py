#date: 2024-11-13T16:58:47Z
#url: https://api.github.com/gists/057683108f6e333078603b4776077883
#owner: https://api.github.com/users/docsallover

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# ... (same as above)

# Create a sequential model using Keras
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)