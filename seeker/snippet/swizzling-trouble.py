#date: 2025-12-03T17:00:20Z
#url: https://api.github.com/gists/199e417bde68126aef53be30426e2127
#owner: https://api.github.com/users/jaro-sevcik

import jax
import jax.numpy as jnp

import jax.random as jr

import scipy

import numpy as np

import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu

from jax._src.interpreters import mlir
from jax.experimental.mosaic import gpu as mgpu
from jaxlib.mlir import ir
from jax.experimental.mosaic.gpu import fragmented_array as fa

from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import gpu as gpu_dialect
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import vector
from jaxlib.mlir.dialects import math

from jax._src import dtypes

jax.config.update("jax_pallas_use_mosaic_gpu", True)


tile_n = 256
def block_max_kernel(x_ref, o_ref, smem_ref, barrier):
  n_idx = jax.lax.axis_index("n")
  n_slice = pl.ds(n_idx * tile_n, tile_n)

  print(x_ref.at[n_slice, :].shape)
  print(smem_ref.shape)

  plgpu.copy_gmem_to_smem(x_ref.at[n_slice, :], smem_ref, barrier)
  plgpu.barrier_wait(barrier)

  layout = plgpu.Layout.TILED(
    plgpu.Tiling(((128, 16), (32, 16), (4,))),
    warp_dims=(-5,),
    lane_dims=(-3,),
    vector_dim=-1
  )

  # ERROR
  # ValueError: Failed to synthesize a transfer pattern that avoids bank conflicts
  # (but it does pass with optimized=False keyword argument)
  x = plgpu.load(smem_ref, (), layout=layout)

  # ... now max reduction and store back ...
  # TODO


n = 1024 * 16
x = jnp.arange(n, dtype=jnp.int32).reshape(-1, 16)

transforms=()
# But this is not supported:
# transforms=(plgpu.TilingTransform((128, 16)),)

gpu_kernel = plgpu.kernel(
    block_max_kernel,
    out_shape=(jax.ShapeDtypeStruct(x.shape, jnp.int32), ),
    compiler_params=plgpu.CompilerParams(approx_math=True),
    grid= (1,),
    grid_names=("n"),
    scratch_shapes = (
      plgpu.SMEM((tile_n, 16), jnp.int32, transforms=transforms),
      plgpu.Barrier(num_arrivals=1),
    )
  )

print(jax.jit(gpu_kernel)(x)[0].reshape(-1, 16))