#date: 2021-11-17T17:09:32Z
#url: https://api.github.com/gists/5716a0cc74203b859a32c3ae46996cbe
#owner: https://api.github.com/users/MingSheng92

def create_model():
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
  model.add(tf.keras.layers.Dropout(0.2))
  
  # Flatten units
  model.add(tf.keras.layers.Flatten())
  # Add a hidden layer with dropout
  model.add(tf.keras.layers.Dense(128, activation="relu"))
  model.add(tf.keras.layers.Dropout(0.5))
  # Add a hidden layer with dropout
  model.add(tf.keras.layers.Dense(128, activation="relu"))
  model.add(tf.keras.layers.Dropout(0.2))
  # an output layer for the full class in your dataset
  model.add(tf.keras.layers.Dense(10, activation="softmax"))
  
  # compile to create the model
  model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, decay=0, momentum=0, nesterov=False),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
  )
  
  return model