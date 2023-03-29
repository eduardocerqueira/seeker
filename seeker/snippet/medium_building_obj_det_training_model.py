#date: 2023-03-29T17:49:29Z
#url: https://api.github.com/gists/6dd3aafeb53a643a6c9739fc85fdd919
#owner: https://api.github.com/users/doleron


model = build_model(tf.keras.layers.Input(shape=(input_size, input_size, 1,)))

model.compile(optimizer=tf.keras.optimizers.Adam(), 
    loss = {'classifier_head' : 'categorical_crossentropy', 'regressor_head' : 'mse' }, 
    metrics = {'classifier_head' : 'accuracy', 'regressor_head' : 'mse' })
    
EPOCHS = 100
BATCH_SIZE = 32

history = model.fit(train_ds,
                    steps_per_epoch=(len(training_files) // BATCH_SIZE),
                    validation_data=validation_ds, validation_steps=1, 
                    epochs=EPOCHS)