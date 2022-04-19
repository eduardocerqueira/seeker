#date: 2022-04-19T16:50:20Z
#url: https://api.github.com/gists/0c5f56ac2b7b5fafb52e7eba97496eaa
#owner: https://api.github.com/users/YankoFelipe

from keras.models import Model
from keras.layers import Input, Lambda, Concatenate

from layer_from_saved_model import LayerFromSavedModel


def example_model(shape, n_classes, tf_path):
    batch_size = shape[0]
    num_frames = shape[1]
    input_layer = Input(shape[1:], batch_size=batch_size)
    outs = []
    for i in range(batch_size):
        for j in range(num_frames):
            out = Lambda(lambda x: x[i, j].reshape((1,) + x[i, j].shape))(input_layer)
            out = LayerFromSavedModel(tf_path)(out)
            out.trainable = False
            outs.append(out)
    out = Concatenate()(outs)
    out = Lambda(lambda x: x.reshape((batch_size, num_frames, int(x.shape[1] / (batch_size * num_frames)))))(out)
    # Add the rest of your layers here!
    model = Model(inputs=input_layer, outputs=out)
    return model
