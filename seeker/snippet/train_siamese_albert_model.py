#date: 2022-06-27T16:58:53Z
#url: https://api.github.com/gists/c7858aa092201313a73b940a9bbcb3e2
#owner: https://api.github.com/users/JanSchm

callbacks = [tf.keras.callbacks.ModelCheckpoint('SiameseTriplet_AlbertBase_epoch{epoch:02d}_val-loss{val_loss:.6f}.hdf5', monitor='val_loss', save_best_only=True, verbose=1),]

# Train the network
history = model.fit(train_gen,
                    steps_per_epoch = len(df_train)//BATCH_SIZE+1,
                    batch_size=None,
                    verbose=1,
                    epochs=25,
                    shuffle=True,
                    validation_data=val_gen,
                    validation_steps=len(df_val)//BATCH_SIZE+1,
                    callbacks=callbacks,                    
                    max_queue_size=3,)