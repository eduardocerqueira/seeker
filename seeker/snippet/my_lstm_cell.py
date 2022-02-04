#date: 2022-02-04T16:46:19Z
#url: https://api.github.com/gists/2836dd11a5482dff3c3cd32165b165f0
#owner: https://api.github.com/users/bastings

# Copyright 2022 Google LLC.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, Tuple
from jax import numpy as jnp
from flax.linen import recurrent
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any

KERNEL_INIT = nn.linear.default_kernel_init
RECURRENT_INIT = nn.initializers.orthogonal

class MyLSTMCell(recurrent.RNNCellBase):
  """A PyTorch-compatible LSTM cell."""
  gate_fn: Callable[..., Any] = nn.sigmoid
  activation_fn: Callable[..., Any] = nn.tanh
  kernel_init: Callable[..., Array] = KERNEL_INIT
  recurrent_kernel_init: Callable[..., Array] = RECURRENT_INIT()
  bias_init: Callable[..., Array] = nn.initializers.zeros
  dtype: Dtype = jnp.float32
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, carry, inputs):
    """Performs a single time step of the cell.

    Args:
      carry: the hidden state of the LSTM cell, a tuple (c, h),
        initialized using `MyLSTMCell.initialize_carry`.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.

    Returns:
      A tuple with the new carry (c', h') and the output (h').
    """
    c, h = carry
    features = h.shape[-1]
    
    # Compute [h_i, h_f, h_g, h_o] at the same time for better performance.
    dense_h = nn.Dense(
        features=features * 4,
        use_bias=True,
        kernel_init=self.recurrent_kernel_init,
        bias_init=self.bias_init,
        name='h', 
        dtype=self.dtype, 
        param_dtype=self.param_dtype)(h)
 
    # Compute [i_i, i_f, i_g, i_o] at the same time for better performance.
    dense_i = nn.Dense(
        features=features * 4,
        use_bias=True,  # dense_h already has a bias, but we follow PyTorch.
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        name='i', 
        dtype=self.dtype, 
        param_dtype=self.param_dtype)(inputs)

    # We sum each h_{i,f,g,o} with each i_{i,f,g,o} already now for performance.
    summed_combined_projections = dense_i + dense_h

    # Split into i = i_i + h_i, f = i_f + h_f, g = i_g + h_h, o = i_o + h_o.
    i, g, f, o = jnp.split(summed_combined_projections, 4, axis=-1)

    i = self.gate_fn(i)
    f = self.gate_fn(f)
    g = self.activation_fn(g)
    o = self.gate_fn(o)

    new_c = f * c + i * g
    new_h = o * self.activation_fn(new_c)
    return (new_c, new_h), new_h

  @staticmethod
  def initialize_carry(rng, batch_dims, size, init_fn=nn.initializers.zeros):
    """initialize the RNN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      batch_dims: a tuple providing the shape of the batch dimensions.
      size: the size or number of features of the memory.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given RNN cell.
    """
    key1, key2 = jax.random.split(rng)
    mem_shape = batch_dims + (size,)
    return init_fn(key1, mem_shape), init_fn(key2, mem_shape)