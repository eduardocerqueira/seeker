#date: 2024-01-26T16:48:54Z
#url: https://api.github.com/gists/8f4f7739938f963964a6c69efe1baca0
#owner: https://api.github.com/users/matteoferla

import warnings, sys
#warnings.simplefilter('default', stream=sys.stdout)


import sys, os

print('sys.version_info', sys.version_info)


def test_tf():
    import tensorflow as tf
    print(tf.__version__)
    assert tf.test.is_built_with_cuda(), 'TF not CUDA build'
    print("CUDA version:", tf.sysconfig.get_build_info()["cuda_version"])
    print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])
    print("CUDA library paths:", tf.sysconfig.get_lib())
    assert tf.config.list_physical_devices('GPU'), 'TF: no CUDA devices'
    print("tf.config.list_physical_devices('GPU')", tf.config.list_physical_devices('GPU'))
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print(c)

def test_jax():
    from jax.lib import xla_bridge
    assert xla_bridge.get_backend().platform != 'cpu', 'Jax XLA: Not CUDA '
    import jax.numpy as jnp
    from jax import random
    key = random.PRNGKey(0)
    x = random.normal(key, (5000, 5000), dtype=jnp.float32)
    print( jnp.dot(x, x.T) )

def test_torch():
    import torch
    assert torch.cuda.is_available(), 'Torch no CUDA available'
    print(torch.cuda.device_count(), torch.cuda.get_device_name(0)), 'Torch no CUDA devices counted'
    print(f'Using CuDNN: {torch.backends.cudnn.enabled} ({torch.backends.cudnn.m.version()})')
    device = torch.device("cuda")
    # Create a random tensor and transfer it to the GPU
    x = torch.rand(5, 3).to(device)
    print("A random tensor:", x)
    y = x * x
    print("After calculation:", y)
    print("Calculated on:", y.device)

def test_openmm():
    from openmm import unit
    import openmm as mm
    import openmm.app as mma
    import sys
    import numpy as np
    
    print(mm.Platform.getPluginLoadFailures())
    print(mm.Platform.getPlatformByName('CUDA').supportsKernels('CUDA'))
    print(mm.Platform.findPlatform('CUDA'))

if __name__ == '__main__':
    for fun in (test_jax, test_tf,  test_torch, test_openmm):
        print(f'\n\n{fun.__name__}')
        try:
            fun()
        except Exception as error:
            print(error.__class__.__name__, error)
            