#date: 2025-12-03T17:00:20Z
#url: https://api.github.com/gists/199e417bde68126aef53be30426e2127
#owner: https://api.github.com/users/jaro-sevcik

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu

from jax._src import dtypes

jax.config.update("jax_pallas_use_mosaic_gpu", True)

tile_m = 16
tile_n = 128
block_size = 16
def block_max_kernel(x_ref, o_ref, smem_ref, barrier):
  n_idx = jax.lax.axis_index("n")
  n_slice = pl.ds(n_idx * tile_n, tile_n)

  plgpu.copy_gmem_to_smem(x_ref.at[:, n_slice], smem_ref, barrier)
  plgpu.barrier_wait(barrier)

  layout = plgpu.Layout.TILED(
    plgpu.Tiling(((16, 128), (16, 32), (16, 1))),
    warp_dims=(-5,),
    lane_dims=(-3,),
    vector_dim=-1
  )

  x = plgpu.load(smem_ref, (), layout=layout, optimized=False)

  # This works:
  #
  # y = jnp.max(x, axis = 0)
  #
  # However, this doesn't:


  # NotImplementedError: Unimplemented primitive in Pallas Mosaic GPU lowering: transpose
  xT = x.mT
  # y = jnp.max(xT, axis = -1)

  # ... take max of each block and store in o_ref
  # TODO

m, n = 16, 4096
input = jnp.arange((m * n), dtype=jnp.int32).reshape(m, n)

transforms=()

gpu_kernel = plgpu.kernel(
    block_max_kernel,
    out_shape=(jax.ShapeDtypeStruct((m, n // block_size), jnp.int32), ),
    compiler_params=plgpu.CompilerParams(approx_math=True),
    grid= (n // tile_n,),
    grid_names=("n",),
    scratch_shapes = (
      plgpu.SMEM((tile_m, tile_n), jnp.int32, transforms=transforms),
      plgpu.Barrier(num_arrivals=1),
    )
  )

print(jax.jit(gpu_kernel)(input)[0])