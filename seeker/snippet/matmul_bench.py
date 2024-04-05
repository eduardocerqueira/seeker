#date: 2024-04-05T16:49:10Z
#url: https://api.github.com/gists/7161235587a2ff51306764fe488b9431
#owner: https://api.github.com/users/garrett361

from time import perf_counter
from typing import Union

import torch

if torch.cuda.is_available():
    assert torch.cuda.is_available()
    from torch import cuda as accel

else:
    import intel_extension_for_pytorch as ipex  # noqa
    from torch import xpu as accel

    device = "xpu"
print(f"Using {device=}. {accel.device_count()=}", flush=True)
DTYPE = torch.bfloat16


def benchmark(
    batch_size: int,
    m: int,
    k: int,
    n: int,
    warmups: int,
    num_iters: int,
    cache_size_MiB: int = 256,
    clear_cache: bool = True,
) -> dict[str, Union[int, float]]:
    """
    Benchmarking m x k by k x n matmuls.
    """
    assert warmups > 0, "Use at least one warmup"
    if clear_cache:
        cache = torch.empty(cache_size_MiB * 2**20, dtype=torch.int8, device=device)
    with torch.inference_mode():
        T = torch.randn(batch_size, m, k, device=device, dtype=DTYPE)
        W = torch.randn(k, n, device=device, dtype=DTYPE)

        for _ in range(warmups):
            T @ W

        # NOTE: @garrett.goon - using Events (as in torch.cuda.Event) is the preferred way to time
        # GPU operations, but the xpu.Event timing gave strange results when 2.1.0a0+cxx11.abi +
        # ipex. TBD whether this is an xpu or code issue.

        total_time = 0.0
        for _ in range(num_iters):
            if clear_cache:
                cache.zero_()
            accel.synchronize()
            start = perf_counter()
            T @ W
            accel.synchronize()
            stop = perf_counter()
            total_time += stop - start
        mean_time_s = total_time / num_iters

        FLOPs = (2 * k - 1) * batch_size * m * n
        elem_size = T.element_size()
        read_write_mem = (n * k + k * m + n * m) * elem_size * batch_size
        del T
        del W
        accel.empty_cache()
        return {
            "time": mean_time_s,
            "comp_intensity": FLOPs / (read_write_mem),
            "TFLOPs": FLOPs / 2**40,
            "TFLOP/s": FLOPs / mean_time_s / 2**40,
            "m": m,
            "n": n,
            "k": k,
        }
