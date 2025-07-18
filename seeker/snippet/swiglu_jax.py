#date: 2025-07-18T17:02:51Z
#url: https://api.github.com/gists/648469815092fc565d2a29a4e3f603af
#owner: https://api.github.com/users/pashu-cohere

from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, random

# -----------------------------------------------------------------------------
# Problem sizes & dtypes
# -----------------------------------------------------------------------------
M    = 1024    
N    = 8192
K    = 28672     

DTYPE_IN  = jnp.bfloat16
DTYPE_ACC = jnp.float32  # accumulate & output

key = random.PRNGKey(0)
key, k1, k2, k3 = random.split(key, 4)

# Inputs ----------------------------------------------------------------------
X   = random.normal(k1, (M, K), dtype=DTYPE_IN)
W1  = random.normal(k2, (K, N), dtype=DTYPE_IN)
W2  = random.normal(k3, (K, N), dtype=DTYPE_IN)
Wcat = jnp.concatenate([W1, W2], axis=1)

print(X.device)  

# -----------------------------------------------------------------------------
# Kernels
# -----------------------------------------------------------------------------

@partial(jax.jit, static_argnums=())
def swiglu_baseline(x: jnp.ndarray, w1: jnp.ndarray, w2: jnp.ndarray) -> jnp.ndarray:
    cand  = x.astype(DTYPE_ACC) @ w1.astype(DTYPE_ACC)
    gate  = jax.nn.sigmoid(x.astype(DTYPE_ACC) @ w2.astype(DTYPE_ACC))
    return (cand * gate)  # cast down for fairness

@partial(jax.jit, static_argnums=())
def swiglu_fused(x: jnp.ndarray, wcat: jnp.ndarray) -> jnp.ndarray:
    y = x.astype(DTYPE_ACC) @ wcat.astype(DTYPE_ACC)
    cand, gate = jnp.split(y, 2, axis=-1)
    return (cand * jax.nn.sigmoid(gate))

# -----------------------------------------------------------------------------
# Benchmark helpers
# -----------------------------------------------------------------------------

def benchmark(fn, *args, warmup: int = 3, steps: int = 30):
    # Ensure compile
    fn(*args).block_until_ready()
    # Warm‑up
    for _ in range(warmup):
        fn(*args).block_until_ready()

    t0 = time.perf_counter()
    for _ in range(steps):
        fn(*args).block_until_ready()
    t_ms = (time.perf_counter() - t0) * 1e3 / steps
    return t_ms

# FLOP counts -----------------------------------------------------------------
FLOPS_PER_MATMUL = 2 * M * K * N  # 2 flops per MAC
FLOPS_EW         = "**********"

baseline_flops = 2 * FLOPS_PER_MATMUL + 2 * FLOPS_EW  # two GEMMs
fused_flops    = 2 * M * K * (2 * N) + 2 * FLOPS_EW  # one big GEMM

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
print("Compiling kernels …")
_ = swiglu_baseline.lower(X, W1, W2).compile()
_ = swiglu_fused.lower(X, Wcat).compile()
print("Compile done. Timing …")

baseline_ms = benchmark(swiglu_baseline, X, W1, W2)
fused_ms    = benchmark(swiglu_fused, X, Wcat)

print("\nResults (H100, BF16 → FP32):")
print(f"Baseline   : {baseline_ms:6.2f} ms | {baseline_flops / (baseline_ms*1e-3) / 1e12:6.1f} TFLOP/s")
print(f"Fused      : {fused_ms:6.2f} ms | {fused_flops    / (fused_ms*1e-3)    / 1e12:6.1f} TFLOP/s")
print(f"Speed‑up    : {baseline_ms / fused_ms:5.2f}× faster\n")
{baseline_ms / fused_ms:5.2f}× faster\n")
