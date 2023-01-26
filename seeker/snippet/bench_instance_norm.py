#date: 2023-01-26T17:05:29Z
#url: https://api.github.com/gists/a8051c150c51f0505d97862557290344
#owner: https://api.github.com/users/jacobhinkle

import apex
import torch
import torch.utils.benchmark as benchmark
from tqdm import tqdm

import csv
import itertools


def get_gpu_name():
    """Query nvidia-smi to find name of GPU"""
    import subprocess

    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
    )
    return proc.stdout.strip().decode("utf-8")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    dtypes = {
        "half": torch.half,
        "bfloat16": torch.bfloat16,
        "float": torch.float,
        "double": torch.double,
    }
    parser.add_argument(
        "--no_pytorch_timer",
        action="store_true",
        help="If given, just run code without repeating using torch.utils.benchmark.Timer. Useful for profiling",
    )
    parser.add_argument(
        "--min_runtime",
        default=0.2,
        type=float,
        help="Minimum amount of time to iterate (ignored if --no_pytorch_timer is given).",
    )
    parser.add_argument(
        "--dtype",
        type=lambda s: dtypes[s],
        nargs="+",
        default=(torch.float, torch.half),
        help="Precision of inputs. At least one of: " + " ".join(dtypes.keys()),
    )
    parser.add_argument(
        "--track_running_stats",
        "-t",
        type=int,
        nargs="+",
        choices=(0, 1),
        default=(0, 1),
        help="Compute running mean and variance",
    )
    parser.add_argument(
        "--affine",
        "-a",
        type=int,
        nargs="+",
        choices=(0, 1),
        default=(0, 1),
        help="Run with weights and biases",
    )
    parser.add_argument(
        "--memory_format",
        "-m",
        nargs="+",
        choices=("contiguous", "channels_last", "strided"),
        default=("contiguous", "channels_last", "strided"),
        help="Memory format for input",
    )
    parser.add_argument(
        "--batch_size", "-B", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="Batch size"
    )
    parser.add_argument(
        "--num_channels",
        "-C",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Num channels",
    )
    parser.add_argument(
        "--spatial_size",
        "-S",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64],
        help="Spatial size",
    )
    parser.add_argument(
        "--implementation",
        "-i",
        nargs="+",
        choices=("eager", "apex", "eager_batchnorm"),
        default=("eager", "apex"),
        help='Which implementations to use: one of "apex", "eager", "eager_batchnorm',
    )
    parser.add_argument(
        "--direction",
        "-d",
        nargs="+",
        choices=("forward", "backward"),
        default=("forward", "backward"),
        help="Directions for evaluation layer",
    )
    parser.add_argument(
        "--output_csv", "-o", default=None, help="If given, write results to a CSV"
    )
    args = parser.parse_args()

    csv_handle = None
    if args.output_csv is not None:
        csv_handle = open(args.output_csv, 'w', newline='')
        csv_writer = csv.DictWriter(
            csv_handle,
            fieldnames=[
                "gpu",
                "direction",
                "dtype",
                "track_running_stats",
                "affine",
                "memory_format",
                "batch_size",
                "num_channels",
                "spatial_size",
                "implementation",
                "num_measurement_runs",
                "num_measurements_per_run",
                "measurement_mean",
                "measurement_median",
                "measurement_iqr",
            ],
        )
        csv_writer.writeheader()
        if args.no_pytorch_timer:
            print(
                "WARNING: CSV output is only written when NOT using --no_pytorch_timer"
            )

    gpu_name = get_gpu_name()

    results = []

    for (
        direction,
        dtype,
        track_running_stats,
        affine,
        mem_format,
        B,
        C,
        S,
        impl,
    ) in tqdm(
        list(
            itertools.product(
                sorted(set(args.direction)),
                set(args.dtype),
                sorted(set(args.track_running_stats)),
                sorted(set(args.affine)),
                sorted(set(args.memory_format)),
                sorted(set(args.batch_size)),
                sorted(set(args.num_channels)),
                sorted(set(args.spatial_size)),
                sorted(set(args.implementation)),
            )
        )
    ):
        track_running_stats = bool(track_running_stats)
        affine = bool(affine)
        bench_label = (
            f"dtype={dtype} "
            f"track_running_stats={track_running_stats} "
            f"affine={affine} "
            f"mem_format={mem_format} "
            f"direction={direction} "
        )
        bench_sub_label = f"shape={B} {C} {S} {S} {S} " f"impl={impl} "
        # print('Benchmark:', bench_label)

        cls = {
            "apex": apex.normalization.instance_norm.InstanceNorm3dNVFuser,
            "eager": torch.nn.InstanceNorm3d,
            "eager_batchnorm": torch.nn.BatchNorm3d,
        }[impl]

        device = torch.device("cuda:0")

        mod = cls(
            C,
            affine=affine,
            track_running_stats=track_running_stats,
            device=device,
            dtype=dtype,
        )

        if mem_format == "contiguous":
            x = torch.randn(
                (B, C, S, S, S),
                dtype=dtype,
                device=device,
            )
        elif mem_format == "channels_last":
            x = torch.randn(
                (B, C, S, S, S),
                dtype=dtype,
                device=device,
            ).to(memory_format=torch.channels_last_3d)
        elif mem_format == "strided":
            x = torch.randn(
                (B, C, 2 * S, S, S),
                dtype=dtype,
                device=device,
            )[:, :, ::2, :, :]

        if direction == "forward":
            mod.eval()
            x.requires_grad_(False)
            _ = mod(x)  # trigger compile before benchmarking
            if args.no_pytorch_timer:
                for _ in range(5):
                    torch.cuda.nvtx.range_push(bench_label + ":" + bench_sub_label)
                    _ = mod(x)
                    torch.cuda.nvtx.range_pop()
            else:
                t = benchmark.Timer(
                    stmt="mod(x)",
                    globals={"mod": mod, "x": x},
                    label=bench_label,
                    sub_label=bench_sub_label,
                    description="raw runtime",
                ).blocked_autorange(min_run_time=args.min_runtime)
                csv_writer.writerow(
                    {
                        "gpu": gpu_name,
                        "dtype": dtype,
                        "track_running_stats": track_running_stats,
                        "affine": affine,
                        "memory_format": mem_format,
                        "direction": direction,
                        "batch_size": B,
                        "num_channels": C,
                        "spatial_size": S,
                        "implementation": impl,
                        "num_measurement_runs": len(t.raw_times),
                        "num_measurements_per_run": t.number_per_run,
                        "measurement_mean": t.mean,
                        "measurement_median": t.median,
                        "measurement_iqr": t.iqr,
                    }
                )
                results.append(t)
        elif direction == "backward":
            mod.train()
            x.requires_grad_(True)
            go = torch.randn_like(x)
            y = mod(x)
            y.backward(go)  # trigger compile before benchmarking
            if args.no_pytorch_timer:
                for p in mod.parameters():
                    p.detach_()
                y = mod(x)
                for _ in range(100):
                    torch.cuda.nvtx.range_push(bench_label + ":" + bench_sub_label)
                    y.backward(go, retain_graph=True)
                    torch.cuda.nvtx.range_pop()
            else:
                t = benchmark.Timer(
                    stmt="y.backward(go, retain_graph=True)",
                    setup="""
                          for p in mod.parameters():
                              p.detach_()
                          y = mod(x)
                          """,
                    globals={"mod": mod, "x": x, "go": go},
                    label=bench_label,
                    sub_label=bench_sub_label,
                    description="raw runtime",
                ).blocked_autorange(min_run_time=args.min_runtime)
                csv_writer.writerow(
                    {
                        "gpu": gpu_name,
                        "dtype": dtype,
                        "track_running_stats": track_running_stats,
                        "affine": affine,
                        "memory_format": mem_format,
                        "direction": direction,
                        "batch_size": B,
                        "num_channels": C,
                        "spatial_size": S,
                        "implementation": impl,
                        "num_measurement_runs": len(t.raw_times),
                        "num_measurements_per_run": t.number_per_run,
                        "measurement_mean": t.mean,
                        "measurement_median": t.median,
                        "measurement_iqr": t.iqr,
                    }
                )
                results.append(t)

    if csv_handle is not None:
        csv_handle.close()

    if not args.no_pytorch_timer:
        compare = benchmark.Compare(results)
        compare.print()
