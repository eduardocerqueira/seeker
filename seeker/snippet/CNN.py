#date: 2021-11-17T16:55:25Z
#url: https://api.github.com/gists/f4fde3e7d70cd6180385450e1d21d733
#owner: https://api.github.com/users/MingSheng92

# function to create Convolutional Neural Network
def createCNN():
        
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.2),

    # Flatten units
    tf.keras.layers.Flatten(),
    # Add a hidden layer with dropout
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    # Add a hidden layer with dropout
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    # an output layer for the full class in your dataset
    tf.keras.layers.Dense(10, activation="softmax")
  ])

  # compile to create the model
  model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, decay=0, momentum=0, nesterov=False),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
  )
  
  return model
    