#date: 2023-07-28T16:59:57Z
#url: https://api.github.com/gists/d4e07a538c661389c005ae93c2cf1a58
#owner: https://api.github.com/users/adalca

import neurite as ne
import keras.backend as K

class DynamicallyWeightedDice:

    def __init__(self, weight_tensor, laplace_smoothing=0.1):
        self.weights = weight_tensor
        self.laplace_smoothing = laplace_smoothing

    def dice(self, y_true, y_pred):
        # NOTE that this does not do any error checking, etc. ys are expected to be in [0, 1] and add up to 1 appropriately
        # ys are expected to be [B, ..., C]

        # reshape to [batch_size, nb_voxels, nb_labels]
        y_true = ne.utils.batch_channel_flatten(y_true)
        y_pred = ne.utils.batch_channel_flatten(y_pred)

        w = ne.utils.batch_channel_flatten(self.weights)

        # compute dice for each entry in batch.
        # dice will now be [batch_size, nb_labels]
        top = 2 * K.sum(w * y_true * y_pred, 1)
        bottom = K.sum(w * K.square(y_true), 1) + K.sum(w * K.square(y_pred), 1)

        # compute Dice
        eps = self.laplace_smoothing
        return (top + eps) / (bottom + eps)