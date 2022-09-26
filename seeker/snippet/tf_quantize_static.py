#date: 2022-09-26T17:12:28Z
#url: https://api.github.com/gists/1c72b2a0d5bbc813ac4f892c21e1d1c4
#owner: https://api.github.com/users/tiandiao123

import tensorflow as tf
import numpy as np

def representative_dataset():
    for _ in range(100):
      data = np.random.rand(1, 224, 224, 3)
      yield [data.astype(np.float32)]


saved_model_dir = "/Users/cuiqingli123/Workspace/torch_op_exp/mobile_net_v3_small"
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_model = converter.convert()

# Save the model.
with open('mobile_net_v3_small_static_quantized.tflite', 'wb') as f:
  f.write(tflite_model)
