#date: 2025-12-03T17:00:20Z
#url: https://api.github.com/gists/199e417bde68126aef53be30426e2127
#owner: https://api.github.com/users/jaro-sevcik

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import time

batch_size = 128
m = 8192
n = 8192
dtype = jnp.bfloat16

key = jax.random.PRNGKey(0)
A = jax.random.normal(key, (batch_size, m, n), dtype=dtype) * 1000.0

@jax.jit
def quantize_fp4_simple(A):
  blocks = A.reshape((-1, 16))
  block_max = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True).astype(jnp.float32)
  scale = (block_max / jnp.finfo(jnp.float4_e2m1fn).max.astype(jnp.float32))
  scale_p = scale
  scale = jax.lax.reduce_precision(scale, 5, 3)
  scale = jnp.clip(scale, 0, jnp.finfo(jnp.float8_e4m3fn).max.astype(jnp.float32))
  scale_inv = 1.0 / scale
  scaled = (blocks * scale_inv).astype(jnp.bfloat16)
  scale = scale.reshape(A.shape[:-1] + (A.shape[-1] // 16,))
  quant = scaled.astype(jnp.float4_e2m1fn)
  return quant.reshape(A.shape), scale.astype(jnp.float8_e4m3fn), scale_p

from jax._src.interpreters import mlir
from jax.experimental.mosaic import gpu as mgpu
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith
from jaxlib.mlir.dialects import gpu
from jaxlib.mlir.dialects import math
from jaxlib.mlir.dialects import memref
from jaxlib.mlir.dialects import llvm
from jaxlib.mlir.dialects import nvvm
from jaxlib.mlir.dialects import vector
from jaxlib.mlir.dialects import scf
from typing import Callable

def get_amax(vec):
  vec = math.absf(vec)
  return vector.reduction(vec.type.element_type, vector.CombiningKind.MAXIMUMF, vec)

def store_scales(scales_to_store, out_scales, scale_idx):
  index = ir.IndexType.get()
  for block_idx in range(scales_to_store.type.shape[0]):
    scale = vector.extract(scales_to_store, dynamic_position=[], static_position=[block_idx])
    scale = vector.broadcast(ir.VectorType.get([1], scales_to_store.type.element_type), scale)
    vector.store(scale, out_scales, [arith.addi(scale_idx, mgpu.c(block_idx, index))])

def convert_bf16_vector_to_fp4(block):
  FP4_PER_INT32 = 8
  i32 = ir.IntegerType.get_signless(32)
  block_size = block.type.shape[0]
  quant_buffer_vector = llvm.mlir_undef(ir.VectorType.get([block_size // FP4_PER_INT32], i32))
  for chunk_idx in range(block_size // FP4_PER_INT32):
    vals = [vector.extract(block, dynamic_position=[], static_position=[chunk_idx * FP4_PER_INT32 + i]) for i in range(FP4_PER_INT32)]
    fp4_values_packed = llvm.inline_asm(
      i32,
      vals,
      ( "{" +
        "  .reg .b8 byte0;\n" +
        "  .reg .b8 byte1;\n" +
        "  .reg .b8 byte2;\n" +
        "  .reg .b8 byte3;\n" +
        "  cvt.rn.satfinite.e2m1x2.f32 byte0, $8, $7;\n" +
        "  cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $5;\n" +
        "  cvt.rn.satfinite.e2m1x2.f32 byte2, $4, $3;\n" +
        "  cvt.rn.satfinite.e2m1x2.f32 byte3, $2, $1;\n" +
        "  mov.b32 $0, {byte3, byte2, byte1, byte0}; }\n"
      ),  
      "=r,f,f,f,f,f,f,f,f",
      has_side_effects=False,
    )
    fp4_values_packed = arith.bitcast(i32, fp4_values_packed)
    quant_buffer_vector = vector.insert(fp4_values_packed, quant_buffer_vector, dynamic_position=[], static_position=[chunk_idx])
  return quant_buffer_vector

def convert_fp8_vector_fp32(scales_fp8):
  fp8 = ir.Float8E4M3FNType.get()
  i16 = ir.IntegerType.get_signless(16)
  fp32 = ir.F32Type.get()

  assert scales_fp8.type.shape[0] % 2 == 0
  assert len(scales_fp8.type.shape) == 1
  assert scales_fp8.type.element_type == fp8

  scales = llvm.mlir_undef(ir.VectorType.get([scales_fp8.type.shape[0]], fp32))
  for block_idx in range(0, scales_fp8.type.shape[0], 2):
    scale0_fp8 = vector.extract(scales_fp8, dynamic_position=[], static_position=[block_idx])
    scale1_fp8 = vector.extract(scales_fp8, dynamic_position=[], static_position=[block_idx + 1])
    packed_f8s = vector.from_elements(ir.VectorType.get([2], fp8), [scale0_fp8, scale1_fp8])
    packed_f8s = vector.bitcast(ir.VectorType.get([1], i16), packed_f8s)
    packed_f8s = vector.extract(packed_f8s, dynamic_position=[], static_position=[0])

    scale_pair_fp8_type = ir.VectorType.get([2], fp8)
    packed = llvm.inline_asm(
      ir.Type.parse("!llvm.struct<(f32,f32)>"),
      [packed_f8s],
      ( "{                                       \n" +
        ".reg .b32 a;                            \n" +
        ".reg .f16 a<2>;                         \n" +
        "cvt.rn.f16x2.e4m3x2 a, $2;               \n" +
        "mov.b32 {a0, a1}, a;                    \n" +
        "cvt.f32.f16 $0, a0;                    \n" +
        "cvt.f32.f16 $1, a1;                    \n" +
        "}" ),
      "=f,=f,h",
      has_side_effects=False,
    )
    scale0 = llvm.extractvalue(fp32, packed, [0])
    scale1 = llvm.extractvalue(fp32, packed, [1])
    
    scales = vector.insert(scale0, scales, dynamic_position=[], static_position=[block_idx])
    scales = vector.insert(scale1, scales, dynamic_position=[], static_position=[block_idx + 1])
    return scales

def convert_fp32_vector_fp8(scales):
  fp8 = ir.Float8E4M3FNType.get()
  i16 = ir.IntegerType.get_signless(16)
  fp32 = ir.F32Type.get()

  assert scales.type.element_type == fp32
  assert len(scales.type.shape) == 1
  assert scales.type.shape[0] % 2 == 0

  scales_fp8_type = ir.VectorType.get([scales.type.shape[0]], fp8)
  scales_fp8 = llvm.mlir_undef(scales_fp8_type)

  for block_idx in range(0, scales.type.shape[0], 2):
    scale0 = vector.extract(scales, dynamic_position=[], static_position=[block_idx])
    scale1 = vector.extract(scales, dynamic_position=[], static_position=[block_idx + 1])
    packed_f8s = llvm.inline_asm(
      i16,
      [scale0, scale1],
      ( "{                                       \n" +
        "cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;  \n" +
        "}" ),
      "=h,f,f",
      has_side_effects=False,
    )

    packed_f8s = vector.broadcast(ir.VectorType.get([1], i16), packed_f8s)
    packed_f8s = vector.bitcast(ir.VectorType.get([2], fp8), packed_f8s)
    scale0_fp8 = vector.extract(packed_f8s, dynamic_position=[], static_position=[0])
    scale1_fp8 = vector.extract(packed_f8s, dynamic_position=[], static_position=[1])
    scales_fp8 = vector.insert(scale0_fp8, scales_fp8, dynamic_position=[], static_position=[block_idx])
    scales_fp8 = vector.insert(scale1_fp8, scales_fp8, dynamic_position=[], static_position=[block_idx + 1])
  return scales_fp8

def quantize_fp4_mosaic(shape: tuple[int,]) -> Callable[[jax.Array], tuple[jax.Array, jax.Array]]:
  FP4_PER_INT32 = 8

  BLOCK_SIZE = 16
  BLOCKS_PER_THREAD = 2
  VALUES_PER_THREAD = BLOCK_SIZE * BLOCKS_PER_THREAD

  THREAD_GRID_BLOCK_SIZE = 128

  assert shape[-1] % BLOCK_SIZE == 0, f"shape {shape} not divisible by {BLOCK_SIZE}"

  def kernel(ctx, x, global_scale, out_quantized, out_scales, _):
    index = ir.IndexType.get()
    fp32 = ir.F32Type.get()
    bf16 = ir.BF16Type.get()

    n = mgpu.c(x.type.shape[0], index)

    block_id = gpu.block_id(gpu.Dimension.x)
    block_dim = gpu.block_dim(gpu.Dimension.x)
    thread_id = gpu.thread_id(gpu.Dimension.x)

    tid = arith.addi(arith.muli(block_dim, block_id), thread_id)
    input_idx = arith.muli(mgpu.c(VALUES_PER_THREAD, index), tid)
    scale_idx = arith.muli(mgpu.c(BLOCKS_PER_THREAD, index), tid)

    fp4_max_inv = arith.constant(fp32, float(1.0 / jnp.finfo(jnp.float4_e2m1fn).max))

    fp32_vector_type = ir.VectorType.get([VALUES_PER_THREAD], fp32)
    global_scale_inv = arith.divf(arith.constant(fp32, 1.0), memref.load(global_scale, []))    
    global_scale_inv = vector.broadcast(fp32_vector_type, global_scale_inv)

    
    # Load a block.
    bf16_vector_type = ir.VectorType.get([VALUES_PER_THREAD], bf16)
    bf16_vector = vector.load(bf16_vector_type, x, [input_idx])

    # Convert to fp32.
    fp32_vector = arith.extf(fp32_vector_type, bf16_vector)
    fp32_vector = arith.mulf(fp32_vector, global_scale_inv)

    fp32_block_type = ir.VectorType.get([BLOCK_SIZE], fp32)

    blocks = []
    for block_idx in range(BLOCKS_PER_THREAD):
      block = vector.extract_strided_slice(fp32_block_type, fp32_vector, [block_idx * BLOCK_SIZE], [BLOCK_SIZE], [1])
      blocks.append(block)

    # Compute scales.
    scales = llvm.mlir_undef(ir.VectorType.get([BLOCKS_PER_THREAD], fp32))
    for block_idx in range(BLOCKS_PER_THREAD):
      # Find the maximum absolute value of each block.
      amax = get_amax(blocks[block_idx])
      scale = arith.mulf(amax, fp4_max_inv)
      scales = vector.insert(scale, scales, dynamic_position=[], static_position=[block_idx])

    # Truncate scales to fp8.
    scales_fp8 = convert_fp32_vector_fp8(scales)
    scales = convert_fp8_vector_fp32(scales_fp8)

    # Scale the blocks.
    for block_idx in range(BLOCKS_PER_THREAD):
      scale = vector.extract(scales, dynamic_position=[], static_position=[block_idx])
      scale_inv = arith.divf(arith.constant(fp32, 1.0), scale)
      blocks[block_idx] = arith.mulf(blocks[block_idx], vector.broadcast(blocks[block_idx].type, scale_inv))

    # Convert blocks to fp4 and store them to the output buffer.
    for block_idx in range(BLOCKS_PER_THREAD):
      block = blocks[block_idx]
      quant_buffer_vector = convert_bf16_vector_to_fp4(block)
      output_quant_idx = arith.addi(arith.muli(mgpu.c(VALUES_PER_THREAD // FP4_PER_INT32, index), tid), 
                                    mgpu.c(block_idx * BLOCK_SIZE // FP4_PER_INT32, index))
      vector.store(quant_buffer_vector, out_quantized, [output_quant_idx])

    store_scales(scales_fp8, out_scales, scale_idx)

  n = 1
  for dim in shape:
    n *= dim

  in_type = jnp.bfloat16
  out_quant_type = jnp.int32  
  out_scale_type = jnp.float8_e4m3fn
  in_shape = jax.ShapeDtypeStruct((n,), in_type), jax.ShapeDtypeStruct((), jnp.float32),
  out_shape = jax.ShapeDtypeStruct((n // FP4_PER_INT32,), out_quant_type), jax.ShapeDtypeStruct((n // BLOCK_SIZE,), out_scale_type),

  with mlir.make_ir_context(), ir.Location.unknown():
    f = mgpu.as_gpu_kernel(
        kernel,
        grid=(n // (THREAD_GRID_BLOCK_SIZE * VALUES_PER_THREAD), 1, 1),  # grid dimensions
        block=(THREAD_GRID_BLOCK_SIZE, 1, 1),  # block dimensions
        in_shape=in_shape,
        out_shape=out_shape,
        smem_scratch_shape=(),
    )
  def quantize_fp4_mosaic_wrapper(x: jax.Array) -> jax.Array:
    assert x.shape == shape
    out = f(x.reshape((n,)), jnp.float32(1.0))
    q = jax.lax.bitcast_convert_type(out[0], jnp.float4_e2m1fn).reshape(x.shape)
    return q, out[1].reshape(x.shape[:-1] + (x.shape[-1] // BLOCK_SIZE,))
  return jax.jit(quantize_fp4_mosaic_wrapper)

quantize_fp4_mosaic_jit = quantize_fp4_mosaic(A.shape)

from jax._src.cudnn.scaled_matmul_stablehlo import quantize, BlockScaleConfig
config = jax.nn.get_scaled_dot_general_config(mode="nvfp4")
quantize_jax = jax.jit(lambda x: quantize(x, config))

print("===== Quantized values =====\n")
print("Mosaic: ", quantize_fp4_mosaic_jit(A)[0].reshape((-1, 16))[:8, :])
print("Simple: ", quantize_fp4_simple(A)[0].reshape((-1, 16))[:8, :])
print("JAX:    ", quantize_jax(A)[0].reshape((-1, 16))[:8, :])

print("===== Scales =====\n")
print("Mosaic: ", quantize_fp4_mosaic_jit(A)[1][0, 0, :16])
print("Simple: ", quantize_fp4_simple(A)[1][0, 0, :16])
print("JAX:    ", quantize_jax(A)[1][0, 0, :16])

# assert jnp.allclose(quantize_fp4_mosaic_jit(A)[0], quantize_jax(A)[0])
assert jnp.allclose(quantize_fp4_mosaic_jit(A)[0], quantize_fp4_simple(A)[0])
assert jnp.allclose(quantize_fp4_mosaic_jit(A)[1], quantize_fp4_simple(A)[1])

for i in range(5):
  # Timing for simple version
  time_start = time.time()
  scaled_simple = quantize_fp4_simple(A)
  jax.block_until_ready(scaled_simple)
  time_end = time.time()
  elapsed_ms_simple = (time_end - time_start) * 1000
  print(f"JAX naive nvfp4 quantize time taken: {elapsed_ms_simple:.2f} ms")

  # Timing for mosaic version
  time_start = time.time()
  scaled_mosaic = quantize_fp4_mosaic_jit(A)
  jax.block_until_ready(scaled_mosaic)
  time_end = time.time()
  elapsed_ms_mosaic = (time_end - time_start) * 1000
  print(f"JAX nvfp4 quantize (mosaic) time taken: {elapsed_ms_mosaic:.2f} ms")

  # Timing for quantize_jax version
  time_start = time.time()
  scaled_quantize = quantize_jax(A)
  jax.block_until_ready(scaled_quantize)
  time_end = time.time()
  elapsed_ms_quantize = (time_end - time_start) * 1000
  print(f"JAX scaled_matmul_stablehlo.quantize time taken: {elapsed_ms_quantize:.2f} ms")
