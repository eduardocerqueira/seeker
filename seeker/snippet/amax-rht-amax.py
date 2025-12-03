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
from jax._src.pallas.mosaic_gpu import pipeline as mgpu_pipeline

from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import gpu as gpu_dialect
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import vector
from jaxlib.mlir.dialects import math

from jax._src import dtypes

import transformer_engine.jax.cpp_extensions.amax as te_amax

jax.config.update("jax_pallas_use_mosaic_gpu", True)

dtype = jnp.bfloat16

def max_abs_mgpu(a, v):
  # Unfortunately, llvm seems to do bad job with max and absf,
  # so we will do it manually (we could still vectorize, though).
  return llvm.inline_asm(
    ir.BF16Type.get(),
    [a, v],
    "abs.bf16 $0, $2;\nmax.NaN.bf16 $0, $1, $0;\n",
    "=h,h,h",
    has_side_effects=True,
  )

def wmma_16x16x16(a, b, amaxes=None):
  def update_amax(amax, vecs):
    for r in vecs:
      for i in range(r.type.shape[0]):
        r_elem = vector.extract(r, dynamic_position=[], static_position=[i])
        # The following line is much slower, so we use the manual ptx version.
        # amax = arith.maximumf(amax, math.absf(r_elem))
        amax = max_abs_mgpu(amax, r_elem)
    return amax

  assert a.shape == (16, 64)
  assert b.shape == (16, 16)

  a_vecs = a.registers.flatten()
  a_regs = [mgpu.utils.bitcast(r, ir.IntegerType.get_signless(32)) for r in a_vecs]
  b_regs = [mgpu.utils.bitcast(r, ir.IntegerType.get_signless(32)) for r in b.registers.flatten()]

  out = mgpu.FragmentedArray.splat(
        mgpu.c(42.0, ir.F32Type.get()),
        shape=a.shape,
        layout=a.layout,
      )
  out_regs = [
    vector.extract(
      reg,
      dynamic_position=[],
      static_position=ir.DenseI64ArrayAttr.get([pos]),
    )
    for reg in out.registers.flatten()
    for pos in range(out.layout.vector_length)
  ]

  zero = mgpu.c(0.0, ir.F32Type.get())
  in_operands = [*a_regs, *b_regs, *([zero] * 8)]

  constraints = ",".join(["=f" for _ in out_regs] + ["r" for _ in in_operands])

  instr = "wmma.mma.sync.aligned.row.row.m16n16k16.f32.bf16.bf16.f32 "
  counter = -1
  def next_idx():
    nonlocal counter
    counter += 1
    return counter
  out_regs_str = "{" + ",".join([f"${next_idx()}" for _ in range(len(out_regs))]) + "}"
  a_regs_str = "{" + ",".join([f"${next_idx()}" for _ in range(len(a_regs))]) + "}"
  b_regs_str = "{" + ",".join([f"${next_idx()}" for _ in range(len(b_regs))]) + "}"
  c_regs_str = "{" + ",".join([f"${next_idx()}" for _ in range(8)]) + "}"
  ptx = f"{instr} {out_regs_str}, {a_regs_str}, {b_regs_str}, {c_regs_str};"

  result_struct_type = ir.Type.parse(f"!llvm.struct<({','.join([str(ir.F32Type.get()) for i in range(8)])})>")

  out_regs_struct = llvm.inline_asm(
    result_struct_type,
    in_operands,
    ptx,
    constraints,
    has_side_effects=False,
  )

  out_regs = [llvm.extractvalue(ir.F32Type.get(), out_regs_struct, [i]) for i in range(len(out_regs))]

  out_vecs = []
  undef = llvm.mlir_undef(ir.VectorType.get((2,), ir.F32Type.get()))
  for first, second in zip(out_regs[::2], out_regs[1::2]):
    vec = llvm.insertelement(undef, first, position=mgpu.c(0, ir.IntegerType.get_signless(32)))
    vec = llvm.insertelement(vec, second, position=mgpu.c(1, ir.IntegerType.get_signless(32)))
    out_vecs.append(vec)
  out_regs = np.asarray(out_vecs, dtype=object).reshape(out.registers.shape)

  transformed = mgpu.FragmentedArray(
    _registers=out_regs,
    _layout=out.layout,
    _is_signed=None,
  ).astype(ir.BF16Type.get())

  if amaxes is not None:
    # Update pre-rht amax.
    amax_regs = amaxes.registers.flatten()
    assert len(amax_regs) == 1
    amax_vec = amax_regs[0]
    amax = vector.extract(amax_vec, dynamic_position=[], static_position=[0])
    amax = update_amax(amax, a_vecs)

    # Update post-rht amax.
    out_vecs = transformed.registers.flatten()
    amax_rht = vector.extract(amax_vec, dynamic_position=[], static_position=[1])
    amax_rht = update_amax(amax_rht, out_vecs)

    amax_vec = vector.insert(amax, amax_vec, dynamic_position=[], static_position=[0])
    amax_vec = vector.insert(amax_rht, amax_vec, dynamic_position=[], static_position=[1])

    amaxes = mgpu.FragmentedArray(
      _registers=np.array([amax_vec], dtype=object).reshape(amaxes.registers.shape),
      _layout=amaxes.layout,
      _is_signed=None,
    ).astype(ir.BF16Type.get())

  return transformed, amaxes

def test_amax(m, n):

  @jax.jit
  def rht_amax_mgpu(x, hadamard):
    tile_m = 16
    tile_n = 4096
    block_n = 1024

    input = x.reshape(tile_m, -1)

    m = input.shape[0]
    n = input.shape[1]

    assert m == tile_m
    assert n % tile_n == 0

    swizzle = 32
    swizzle_elems = 8 * swizzle // dtypes.bit_width(dtype)

    transforms = (
        plgpu.TilingTransform((16, swizzle_elems)), plgpu.SwizzleTransform(swizzle)
    )

    BLOCK_LAYOUT = plgpu.Layout.TILED(
      plgpu.Tiling(((16, 64), (16, 16), (16, 8), (8, 8), (2,))),
      warp_dims=(-8,),
      lane_dims=(-3, -2),
      vector_dim=-1
    )

    N_PREFETCH = 2
      
    def kernel(x, hadamard, o_amax, x_smems, hadamard_smem, barriers_x, barrier_hadamard):
      n_idx = jax.lax.axis_index("n")

      # Start prefetching blocks to smem.
      for i in range(N_PREFETCH):
        n_slice = pl.ds(n_idx * tile_n + i * block_n, block_n)
        plgpu.copy_gmem_to_smem(x.at[:, n_slice], x_smems[i], barriers_x[i])

      plgpu.copy_gmem_to_smem(hadamard, hadamard_smem, barrier_hadamard)
      plgpu.barrier_wait(barrier_hadamard)

      # Load the hadamard matrix from the shared memory to a fragmented array.
      hadamard_layout = plgpu.Layout.TILED(
        plgpu.Tiling(((16, 16), (8, 16), (8, 8), (2,))),
        warp_dims=(fa.Replicated(4),),
        lane_dims=(-3, -2),
        vector_dim=-1,)
      had = plgpu.load(hadamard_smem, (), layout=hadamard_layout)

      # Initialize the amaxes fragmented array (two amaxes per thread - normal and RHT).
      amaxes_layout = plgpu.Layout.TILED(
        plgpu.Tiling(((128, 2), (32, 2))),
        warp_dims=(-4,),
        lane_dims=(-2,),
        vector_dim=-1,
      )
      amaxes_shape = plgpu.ShapeDtypeStruct((128, 2), dtype, layout=amaxes_layout)
      amaxes = plgpu.layout_cast(jnp.full((128, 2), 0.0, dtype), amaxes_layout)

      # Perform the RHT on the input, collecting the amaxes on the way.
      @plgpu.inline_mgpu(
        arg_types=(BLOCK_LAYOUT, hadamard_layout, amaxes_layout),
        return_type=amaxes_shape,
      )
      def rht(ctx, x_array, hadamard, amaxes):
        for i in range(x_array.shape[0] // 16):
          for j in range(x_array.shape[1] // 64):
            i_slice = slice(i * 16, (i + 1) * 16)
            j_slice = slice(j * 64, (j + 1) * 64)
            _, amaxes = wmma_16x16x16(x_array[i_slice, j_slice], hadamard, amaxes)
        return amaxes

      # TODO Do this with emit_pipeline_warp_specialized (with carry). Ugh.
      n_iters = tile_n // block_n
      for i in range(n_iters):
        # Make sure the tile is loaded into smem.
        plgpu.barrier_wait(barriers_x[i % N_PREFETCH])

        # Load the tile from smem to a fragmented array.
        x_array = plgpu.load(x_smems[i % N_PREFETCH], (), layout=BLOCK_LAYOUT)

        # Kick off loading the next tile into smem.
        if i < n_iters - N_PREFETCH:
          n_slice = pl.ds(n_idx * tile_n + (i + N_PREFETCH) * block_n, block_n)
          plgpu.copy_gmem_to_smem(x.at[:, n_slice], x_smems[i % N_PREFETCH], barriers_x[i % N_PREFETCH])
        amaxes = rht(x_array, had, amaxes)

      o_amax[n_idx] = amaxes

    grid = (n // tile_n, )
    gpu_kernel = plgpu.kernel(
        kernel,
        out_shape=(jax.ShapeDtypeStruct(grid + (128, 2,), dtype), ),
        compiler_params=plgpu.CompilerParams(approx_math=True),
        grid=grid,
        grid_names=("n"),
        num_threads=1,
        thread_name="wg",
        scratch_shapes = (
          # Input matrix tile.
          (plgpu.SMEM((tile_m, block_n), dtype, transforms=transforms),) * N_PREFETCH,
          # Hadamard matrix.
          plgpu.SMEM((16, 16), dtype, transforms=transforms),
          # Barriers for the input matrix tile.
          (plgpu.Barrier(num_arrivals=1),) * N_PREFETCH,
          # Barrier for the hadamard matrix.
          plgpu.Barrier(num_arrivals=1),
        ),
      )
    
    amax_grid, = gpu_kernel(input, hadamard)
    return jnp.max(amax_grid, axis = (0, 1))


  s = jnp.array([1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1], dtype = jnp.int32)
  H = jnp.diag(s) @ jnp.array(scipy.linalg.hadamard(16))
  H =  (H / jnp.sqrt(16)).astype(jnp.bfloat16)

  @jax.jit
  def ref(x):
    amax = jnp.max(jnp.abs(x))
    transformed = (x.reshape(-1, 16) @ H).reshape(x.shape)
    amax_rht = jnp.max(jnp.abs(transformed))
    return jnp.array([amax, amax_rht])

  @jax.jit
  def ref_amax(x):
    return jnp.max(jnp.abs(x))

  @jax.jit
  def ref_amax_te(x):
    return te_amax.calculate_post_rht_amax(x, te_amax.AmaxScope.LOCAL, False, True, -1)

  x = jax.random.normal(jax.random.key(0), (m, n), dtype=dtype)
  amaxes = rht_amax_mgpu(x, H.mT)

  print(ref(x))
  print(rht_amax_mgpu(x, H.mT))
  print(ref_amax_te(x))

  # te_amax.calculate_post_rht_amax(jnp.ones((128, 4096, 4096), dtype=jnp.bfloat16), te_amax.AmaxScope.LOCAL, False, True, 1)

  if not jnp.allclose(amaxes, ref(x), atol=1e-3):
    print("Amax mismatch!!!!!!!!!!!!!!!!!!!!!!!!!")
  else:
    print("Correctness check passed")

  import time

  # Timing the gpu_kernel and ref inside the function
  # Warmup JIT for both functions
  _ = rht_amax_mgpu(x, H.mT)
  _ = ref(x)

  for i in range(10):
    # Time gpu_kernel
    start = time.time()
    result = rht_amax_mgpu(x, H.mT)
    jax.block_until_ready(result)
    end = time.time()
    print(f"gpu_kernel time: {(end - start) * 1000:.2f} ms")

    # Time ref
    start = time.time()
    ref_result = ref(x)
    jax.block_until_ready(ref_result)
    end = time.time()
    print(f"ref time: {(end - start) * 1000:.2f} ms")

    # Time ref_amax
    start = time.time()
    ref_amax_result = ref_amax(x)
    jax.block_until_ready(ref_amax_result)
    end = time.time()
    print(f"ref_amax time: {(end - start) * 1000:.2f} ms")

    # Time ref_amax_te
    start = time.time()
    ref_amax_te_result = ref_amax_te(x)
    jax.block_until_ready(ref_amax_te_result)
    end = time.time()
    print(f"ref_amax_te time: {(end - start) * 1000:.2f} ms")

test_amax(128 * 8192, 8192)

