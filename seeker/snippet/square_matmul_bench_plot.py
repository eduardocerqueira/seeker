#date: 2024-04-05T16:49:10Z
#url: https://api.github.com/gists/7161235587a2ff51306764fe488b9431
#owner: https://api.github.com/users/garrett361

import argparse

import pandas as pd
import seaborn as sns
import torch
import tqdm

from matmul_bench import benchmark

sns.set_theme(style="darkgrid", rc={"figure.figsize": (10, 10)})

"""
Script for benchmarking square matmuls and creating a plot.
"""


def code_by_divisibility(n, max_exp=4):
    """Returns the largest factor = 2 ** exp that n is divisible by, for n in {0, ..., max_exp}."""
    for exp in reversed(range(max_exp + 1)):
        factor = 2**exp
        num_elements = n.k
        if not num_elements % factor:
            return factor


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
    results = []

    for dim in tqdm.trange(args.min_dim, args.max_dim + 1, args.step):
        result = benchmark(
            batch_size=1,
            m=dim,
            k=dim,
            n=dim,
            warmups=args.warmups,
            num_iters=args.num_iters,
            cache_size_MiB=args.cache_size_MiB,
            clear_cache=not args.no_clear_cache,
        )
        results.append(result)

    results_df = pd.DataFrame(results)

    results_df["divisible_by"] = results_df.apply(code_by_divisibility, axis=1)

    plot = sns.scatterplot(x="m", y="TFLOP/s", data=results_df, hue="divisible_by", palette="Set2")
    device = "cuda" if torch.cuda.is_available() else "xpu"
    plot.set(title=f"Square matrix multiplies (batch_size = 1, {device=})")
    plot.set(xlabel="dim")
    plot.figure.savefig(f"flops_vs_dim_{device}.png", dpi=256)


if __name__ == "__main__":
    main()
