#date: 2025-12-03T17:00:20Z
#url: https://api.github.com/gists/199e417bde68126aef53be30426e2127
#owner: https://api.github.com/users/jaro-sevcik

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu

from jax._src import dtypes

jax.config.update("jax_pallas_use_mosaic_gpu", True)

tile_m = 128
tile_n = 128
block_size = 16
def block_max_kernel(x_ref, o_ref, smem_ref, barrier):
  m_idx = jax.lax.axis_index("m")
  n_idx = jax.lax.axis_index("n")
  m_slice = pl.ds(m_idx * tile_m, tile_m)
  n_slice = pl.ds(n_idx * tile_n, tile_n)

  plgpu.copy_gmem_to_smem(x_ref.at[m_slice, n_slice], smem_ref, barrier)
  plgpu.barrier_wait(barrier)

  layout = plgpu.Layout.TILED(
    plgpu.Tiling(((128, 16), (32, 16))),
    warp_dims=(-4,),
    lane_dims=(-2,),
    vector_dim=-1
  )

  x = plgpu.load(smem_ref, (), layout=layout, optimized=False)

  # ERROR
  # NotImplementedError: Unimplemented primitive in Pallas Mosaic GPU lowering: abs
  x = jnp.abs(x)

  # ... take max of each block and store in o_ref
  # TODO

m, n = 1024, 1024
input = jnp.arange((m * n), dtype=jnp.int32).reshape(m, n)

transforms=()

gpu_kernel = plgpu.kernel(
    block_max_kernel,
    out_shape=(jax.ShapeDtypeStruct((m, n // block_size), jnp.int32), ),
    compiler_params=plgpu.CompilerParams(approx_math=True),
    grid= (m // tile_m, n // tile_n),
    grid_names=("m", "n"),
    scratch_shapes = (
      plgpu.SMEM((tile_m, tile_n), jnp.int32, transforms=transforms),
      plgpu.Barrier(num_arrivals=1),
    )
  )

print(jax.jit(gpu_kernel)(input)[0])