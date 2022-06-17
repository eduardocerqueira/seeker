#date: 2022-06-17T16:50:30Z
#url: https://api.github.com/gists/b936a379617cc42afa98cd6582a44168
#owner: https://api.github.com/users/SmiffyKMc

conv_base = keras.applications.vgg16.VGG16(
    weights="imagenet",
    include_top=False
)
conv_base.trainable = False

inputs = keras.Input(shape=(256, 256, 3))
x = data_augmentation(inputs)
x = keras.applications.vgg16.preprocess_input(x)
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(512)(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation=keras.activations.sigmoid)(x)
model = keras.Model(inputs, outputs)

model.compile(loss=keras.losses.BinaryCrossentropy(),
              optimizer=keras.optimizers.RMSprop(),
              metrics=["accuracy"])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=f"{hotDogDir}hotdog_classifier_v4.keras",
        save_best_only=True,
        monitor="val_loss"
    )
]

history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=callbacks
)