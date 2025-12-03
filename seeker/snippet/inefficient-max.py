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
block_size = 16
def block_max_kernel(x_ref, o_ref, smem_ref, barrier):
  m_idx = jax.lax.axis_index("m")
  m_slice = pl.ds(m_idx * tile_m, tile_m)

  plgpu.copy_gmem_to_smem(x_ref.at[m_slice, :], smem_ref, barrier)
  plgpu.barrier_wait(barrier)

  layout = plgpu.Layout.TILED(
    plgpu.Tiling(((128, 16), (32, 16))),
    warp_dims=(-4,),
    lane_dims=(-2,),
    vector_dim=-1
  )

  x = plgpu.load(smem_ref, (), layout=layout, optimized=False)

  x = jnp.reshape(x, (-1, block_size))

  # The following generates terrible code for the max reduction.
  # Is there a way to avoid the nan checks?
  # Here is an excerpt for a couple of max operations:
  #
  # mov.b32         {%rs41, %rs42}, %r21;
  # setp.le.bf16    %p35, %rs40, %rs41;
  # setp.nan.bf16   %p36, %rs41, %rs41;
  # selp.b16        %rs43, %rs41, %rs40, %p35;
  # selp.b16        %rs44, %rs41, %rs43, %p36;
  # setp.le.bf16    %p37, %rs44, %rs42;
  # setp.nan.bf16   %p38, %rs42, %rs42;
  # selp.b16        %rs45, %rs42, %rs44, %p37;
  # selp.b16        %rs46, %rs42, %rs45, %p38;
  x = jnp.maximum.reduce(x, axis = -1)

  o_ref[m_slice] = x

m, n = 1024, 16
input = jnp.arange((m * n), dtype=jnp.bfloat16).reshape(m, n)

transforms=()

gpu_kernel = plgpu.kernel(
    block_max_kernel,
    out_shape=(jax.ShapeDtypeStruct((m,), jnp.bfloat16), ),
    compiler_params=plgpu.CompilerParams(approx_math=True),
    grid= (m // tile_m,),
    grid_names=("m",),
    scratch_shapes = (
      plgpu.SMEM((tile_m, block_size), jnp.bfloat16, transforms=transforms),
      plgpu.Barrier(num_arrivals=1),
    )
  )

print(jax.jit(gpu_kernel)(input)[0])