#date: 2022-12-30T17:07:48Z
#url: https://api.github.com/gists/ce515ff0efc2f7c921d77a0d00dc814a
#owner: https://api.github.com/users/PhillCli

import numpy as np
try:
    import tensorflow as tf
except ImportError:
    tf = None
from time import perf_counter


def timeit(func, *args, **kwargs):
    durations = []
    for i in range(10):
        tic = perf_counter()
        func(*args, **kwargs)
        toc = perf_counter()
        durations.append(toc - tic)
    durations.sort()
    return np.mean(durations[:-5])  # trim stragglers


for dtype in (np.float32, np.float64):
    dtype_name = dtype.__name__
    rng = np.random.default_rng(42)
    A = rng.normal(size=(1024, 1024)).astype(dtype)
    B = rng.normal(size=(1024, 1024)).astype(dtype)
    C = np.empty_like(A @ B)

    d = timeit(np.dot, A, B, out=C)
    GFLOP = A.shape[0] * B.shape[1] * (2 * A.shape[1] + 2) / 1e9
    print(f"[{dtype_name}] np.dot: {d * 1e3:.3f} ms, {GFLOP / d:.1f} GFLOP/s")

    d = timeit(np.linalg.svd, A)
    print(f"[{dtype_name}] np.linalg.svd: {d * 1e3:.3f} ms")

    if tf is not None:
        A = tf.constant(A)
        B = tf.constant(B)

        d = timeit(np.matmul, A, B)
        GFLOP = A.shape[0] * B.shape[1] * (2 * A.shape[1] + 2) / 1e9
        print(f"[{dtype_name}] tf.matmul: {d * 1e3:.3f} ms, {GFLOP / d:.1f} GFLOP/s")

        d = timeit(tf.linalg.svd, A)
        print(f"[{dtype_name}] tf.linalg.svd: {d * 1e3:.3f} ms")
