#date: 2022-07-29T16:54:10Z
#url: https://api.github.com/gists/45a797e72cc31a5bca9a031630586959
#owner: https://api.github.com/users/seabnavin19

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=5, # how many epochs to wait before stopping
    restore_best_weights=True,
)