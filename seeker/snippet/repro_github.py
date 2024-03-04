#date: 2024-03-04T17:06:50Z
#url: https://api.github.com/gists/9fd4022c3eae431120863cf2285edb76
#owner: https://api.github.com/users/lbortolotti

import os
import numpy as np
import keras
from keras import layers
from keras import initializers
from keras import backend as K

# Setting float64 as default dtype removes the discrepancy between CPU and GPU!
# keras.backend.set_floatx('float64')
import tensorflow as tf
from plotly import graph_objects as go

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


theta = np.linspace(0, 2 * np.pi, 1000).reshape(1, -1)

np.random.seed(42)
tf.random.set_seed(42)
dummy_input_dict = {
    "input_a": 1000
    * np.stack((np.cos(theta), np.sin(theta)), axis=-1).astype(np.float32),
    "input_b": np.random.rand(1, 1000, 5).astype(np.float32),
}


def build_model():
    input_a = layers.Input(shape=(1000, 2), name="input_a")
    input_b = layers.Input(shape=(1000, 5), name="input_b")

    x = layers.Concatenate()([input_a, input_b])
    for idx in range(3):
        lstm_layer = layers.LSTM(
                1024,
                kernel_initializer=initializers.RandomNormal(seed=42 + idx),
                recurrent_initializer=initializers.RandomNormal(seed=52 + idx),
                return_sequences=True,
            )
        x = lstm_layer(x)
    y = layers.Dense(1)(x)
    model = keras.Model(inputs=[input_a, input_b], outputs=y)

    return model


def main(device):
    with tf.device(device):
        model = build_model()
        model.load_weights("my_initial_weights.h5")

        features = ["input_a", "input_b"]
        dummy_input = [dummy_input_dict[k] for k in features]
        preds = model.predict(dummy_input)

        intermediate_outputs = []
        intermediate_weights = []

        # Extract weights + intermediate outputs, for plotting
        for idx in range(2, 5):
            get_layer_output = K.function(
                [model.layers[0].input, model.layers[1].input],
                [model.layers[idx].output],
            )

            layer_output = get_layer_output(dummy_input)
            if len(layer_output) == 1:
                layer_output = layer_output[0]
                intermediate_outputs.append(layer_output)
                intermediate_weights.append(model.layers[idx].get_weights())

    return preds, intermediate_outputs, intermediate_weights


if __name__ == "__main__":

    # Save one set of weights, so that we can compare the weights of the two models
    model = build_model()
    model.save_weights("my_initial_weights.h5")

    cpu_preds, cpu_ir, cpu_weights = main("/device:CPU:0")
    gpu_preds, gpu_ir, gpu_weights = main("/device:GPU:0")

    cpu_output = cpu_preds[0, :, 0]
    gpu_output = gpu_preds[0, :, 0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=cpu_output-gpu_output, name="CPU minus GPU"))
    fig.write_html("cpu_vs_gpu.html")

    for i_layer, (this_cpu_ir, this_gpu_ir, this_cpu_w, this_gpu_w) in enumerate(
        zip(cpu_ir, gpu_ir, cpu_weights, gpu_weights)
    ):
        fig = go.Figure()
        # histograms
        fig.add_trace(go.Histogram(x=this_cpu_ir.flatten(), name="CPU"))
        fig.add_trace(go.Histogram(x=this_gpu_ir.flatten(), name="GPU"))
        fig.write_html(f"cpu_vs_gpu_intermediate_layer_{i_layer}.html")

        # histograms
        for i in range(len(this_cpu_w)):
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=this_cpu_w[i].flatten(), name=f"CPU"))
            fig.add_trace(go.Histogram(x=this_gpu_w[i].flatten(), name=f"GPU"))
            fig.write_html(f"cpu_vs_gpu_intermediate_weights_{i_layer}_{i}.html")
