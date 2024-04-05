#date: 2024-04-05T16:49:10Z
#url: https://api.github.com/gists/7161235587a2ff51306764fe488b9431
#owner: https://api.github.com/users/garrett361

import argparse

from matmul_bench import benchmark

"""
Script for benchmarking square matmuls and printing out results.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-dim", default=512, type=int)
    parser.add_argument("--max-dim", default=4096, type=int)
    parser.add_argument("--step", default=256, type=int)
    parser.add_argument("--warmups", default=3, type=int)
    parser.add_argument("--num-iters", default=10, type=int)
    parser.add_argument("--cache-size-MiB", default=256, type=int)
    parser.add_argument("--no-clear-cache", action="store_true")
    args = parser.parse_args()
    for dim in range(args.min_dim, args.max_dim + 1, args.step):
        results = benchmark(
            batch_size=1,
            m=dim,
            k=dim,
            n=dim,
            warmups=args.warmups,
            num_iters=args.num_iters,
            cache_size_MiB=args.cache_size_MiB,
            clear_cache=not args.no_clear_cache,
        )
        print(f"{dim}x{dim} square matmul TFLOP/s: {results=}", flush=True)


if __name__ == "__main__":
    main()
