#date: 2022-06-01T17:12:12Z
#url: https://api.github.com/gists/6d848dc4979113ad3b0fb88ba33c0847
#owner: https://api.github.com/users/yardenzaki

# imports:
import tensorflow as tf

## -------- Using TensorBoard for Model Visualization  -------------------------------------------
# Load the Tensorboard notebook extension
# And import datetime

%load_ext tensorboard
!rm -rf ./logs/ #And we clear all logs from all runs we did before.
# Getting the model
fashion_classifier = classifier()
# Create a callback
tfboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
fashion_classifier.fit(fashion_train, fashion_train_label, epochs=20, validation_split=0.15, callbacks=[tfboard_callback])
%tensorboard --logdir logs