#date: 2023-09-26T16:56:38Z
#url: https://api.github.com/gists/3ee77e8f53244641e56ba9fe7d894749
#owner: https://api.github.com/users/yhtang

import numpy as np
import os, re
import jax
from jax.experimental import maps
from jax.experimental import pjit
import jax.numpy as jnp

from jax.experimental import mesh_utils

from absl import flags
import jax
from jax import numpy as jnp

flags.FLAGS.xla_enable_async_collective_permute = True
flags.FLAGS.xla_tpu_enable_all_experimental_scheduler_features = True

P = pjit.PartitionSpec
# devices = mesh_utils.create_device_mesh((8,))
devices = np.array(jax.devices())
mesh = Mesh(devices, axis_names=('x'))

M = 2048
K = 512
N = 1024

def make(shape):
  return jnp.arange(np.prod(shape)).reshape(shape)
A = make((M, K))
B = make((K, N))
C = A @ B

A_x = jax.device_put(A, NamedSharding(mesh, P('x', None)))
# jax.debug.visualize_array_sharding(A_x)
# A_x.addressable_shards
# print(C)

def collective_matmul_allgather_lhs_non_contracting(lhs, rhs):
  # lhs is the looped operand; rhs is the local operand
  axis_size = jax.lax.psum(1, axis_name='x')
  axis_index = jax.lax.axis_index(axis_name='x')
  chunk_size = lhs.shape[0]

  def f(i, carrys):
    accum, lhs = carrys
    # compute is a simple matmul
    update = lhs @ rhs
    # circular shift to the left
    lhs = jax.lax.ppermute(
        lhs,
        axis_name='x',
        perm=[(j, (j - 1) % axis_size) for j in range(axis_size)]
    )
    # device 0 computes chunks 0, 1, ...
    # device 1 computes chunks 1, 2, ...
    update_index = (((axis_index + i) % axis_size) * chunk_size, 0)
    accum = jax.lax.dynamic_update_slice(accum, update, update_index)
    return accum, lhs

  accum = jnp.zeros((lhs.shape[0] * axis_size, rhs.shape[1]), dtype=lhs.dtype)
  # FIXME
  # fori_loop cause a crash: hlo_sharding.cc:817 Check failed: !IsManual()
  # accum, lhs = jax.lax.fori_loop(0, axis_size - 1, f, (accum, lhs))
  for i in range(0, axis_size - 1):
    # print(f'i={i}, accum={accum}')
    accum, lhs = f(i, (accum, lhs))

  # compute the last chunk, without the ppermute
  update = lhs @ rhs
  i = axis_size - 1
  update_index = (((axis_index + i) % axis_size) * chunk_size, 0)
  accum = jax.lax.dynamic_update_slice(accum, update, update_index)

  return accum  

from jax.experimental.pjit import pjit

def f(a, b):
  return a @ b

pjitted_f = pjit(f, in_axis_resources=(P('x', None), P()), out_axis_resources=P())

# See *blocking* allgather in xprof
with mesh:
  C = pjitted_f(A_x, B)

# see allgather in hlo
with mesh:
  print(pjitted_f.lower(A_x, B).compile().as_text())

jit_sharded_f = jax.jit(shard_map(collective_matmul_allgather_lhs_non_contracting, mesh,
          in_specs=(P('x', None), P()), out_specs=P()))

C = jit_sharded_f(A_x, B)