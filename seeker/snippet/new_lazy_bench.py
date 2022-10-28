#date: 2022-10-28T17:12:06Z
#url: https://api.github.com/gists/a4743179594f3fb29a0c70d9df55e796
#owner: https://api.github.com/users/JackCaoG

import argparse
import csv
import functools
import gc
import io
import itertools
import logging
import numpy as np
import os
import re
import sys
import time
import torch
from torch import nn
from torch.jit import fuser, optimized_execution
from os.path import abspath
from scipy.stats import ttest_ind
import importlib
import glob
import collections
import random
import torch._lazy
import torch._lazy.metrics as metrics
import torch._lazy.ts_backend

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met


def set_seeds(seed=1337):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def get_unique_suffix():
    return f"{time.time()}_{os.getpid()}"

os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam

log = logging.getLogger(__name__)

# Models that are known to crash or otherwise not work with lazy tensor are
# disabled, but should be removed from these lists once fixed
SKIP = {
    "densenet121": "Disabled by torchbench upstream due to OOM on T4 CI machine",
    "timm_nfnet": "Disabled by torchbench upstream due to OOM on T4 CI machine",
    "moco": "Distributed/ProcessGroupNCCL: Tensors must be CUDA and dense",
    "tacotron2": "Disabled by torchbench upstream due to OOM on T4 CI machine",
}
SKIP_TRAIN_ONLY = {
    "squeezenet1_1": "Disabled by torchbench upstream due to OOM on T4 CI machine",
    "demucs": "Disabled by torchbench upstream due to OOM on T4 CI machine",
}

current_name = ""
current_device = ""

@functools.lru_cache(maxsize=None)
def output_csv(name, headers):
    output = csv.writer(
        io.TextIOWrapper(
            open(name, "wb", buffering=0),
            "utf-8",
            write_through=True,
        ),
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL
    )
    output.writerow(headers)
    return output

def pick_grad(args, name):
    if args.test == 'train':
        return torch.enable_grad()

    if name in ("maml",):
        return torch.enable_grad()
    else:
        return torch.no_grad()

def short_name(name, limit=20):
    """Truncate a model name to limit chars"""
    return name if len(name) <= limit else f"{name[:limit - 3].rstrip('_')}..."

def iter_torchbench_model_names():
    from torchbenchmark import _list_model_paths
    for model_path in _list_model_paths():
        model_name = os.path.basename(model_path)
        yield model_name

def iter_models(args, dirpath):
    for name in itertools.chain(iter_toy_model_names(), iter_torchbench_model_names()):
        if (
            (len(args.filter) and (not re.search("|".join(args.filter), name, re.I)))
            or (len(args.exclude) and re.search("|".join(args.exclude), name, re.I))
        ):
            save_error(name, args.test, "disabled via cmdline filter/exclude", dirpath)
            continue
        if name in SKIP:
            save_error(name, args.test, f"SKIP because {SKIP[name]}", dirpath)
            continue
        if name in SKIP_TRAIN_ONLY and args.test == "train":
            save_error(name, args.test, f"SKIP_TRAIN_ONLY because {SKIP_TRAIN_ONLY[name]}", dirpath)
            continue
        yield name

def call_model_with(model, inputs):
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        return model(*inputs)
    elif isinstance(inputs, dict):
        return model(**inputs)
    elif isistance(inputs, torch.Tensor):
        return model(inputs)
    raise RuntimeError("invalid example inputs ", inputs)

class LazySync:
    def __init__(self, sync_every_iter=False, skip_final_sync=False):
        self.sync_every_iter = sync_every_iter
        self.skip_final_sync = skip_final_sync

    def iter_sync(self):
        torch._lazy.mark_step()
        if self.sync_every_iter:
            xm.wait_device_ops()

    def final_sync(self):
        xm.mark_step()
        if self.skip_final_sync:
            return
        xm.wait_device_ops()

def dump_lazy_metrics(reset=False):
    metrics = {name: int(met.counter_value(name)) for name in met.counter_names() if int(met.counter_value(name) > 0)}
    # if reset:
    #     metrics.reset()
    return metrics

def timed(args, benchmark, sync, times=1):
    results = None
    sync.final_sync()
    set_seeds()
    if args.test == 'eval':
        model, example_inputs = benchmark.get_module()

    # keep the lazy tensor results alive until the final sync
    t0 = time.perf_counter()
    for i in range(times):
        if args.test == 'eval':
            results = call_model_with(model, example_inputs)
        elif args.test == 'train':
            benchmark.train()

        # for the last i, let final_sync take care of it
        if i < times - 1:
            # may be just an async 'mark_step' for lazy, or no-op for cuda
            sync.iter_sync()

    # should be a hard sync for lazy and cuda
    # unless strictly measuring lazy trace overhead, then no-op
    sync.final_sync()
    t1 = time.perf_counter()
    return results, t1 - t0

def to_device(tensors, device):
    """Handles moving tensor or tensors (in various containers) to a new device.
        Used for various purposes (either correctness checking, or even as an impromptu
        means of synchronization.) Note: this method doesn't apply a cuda sync, do that outside.
    """

    try:
        import transformers.modeling_outputs
        if (
            isinstance(tensors, transformers.modeling_outputs.MaskedLMOutput) or
            isinstance(tensors, transformers.modeling_outputs.Seq2SeqLMOutput)
        ):
            # huggingface transformers return classes as model output with many attributes
            # we don't want to sync (such as hidden states of every layer) - just sync the logits
            tensors = tensors.logits
    except ImportError:
        pass

    try:
        import torchbenchmark.models.soft_actor_critic.nets
        import torchbenchmark.models.drq.utils
        if (
            isinstance(tensors, torchbenchmark.models.soft_actor_critic.nets.SquashedNormal) or
            isinstance(tensors, torchbenchmark.models.drq.utils.SquashedNormal)
        ):
            # a SquashedNormal is a py class that holds a loc and scale torch tensor,
            # so convert it to a tuple for compatibility with downstream check_results
            tensors = (tensors.loc, tensors.scale)
    except ImportError:
        pass

    if isinstance(tensors, tuple) or isinstance(tensors, list):
        return tuple(to_device(i, device) for i in tensors)
    elif isinstance(tensors, dict):
        return {k: to_device(tensors[k], device) for k in tensors}
    elif isinstance(tensors, torch.Tensor):
        return tensors.to(device)
    raise RuntimeError("invalid example tensors ", tensors)

def lazy_experiment(args, results, benchmark, lazy_benchmark):
    timings = np.zeros((args.repeat, 2), np.float64)
    warmup0 = time.perf_counter()
    for rep in range(args.warmup):
        # interleave the runs to handle frequency scaling and load changes
        #timed(args, benchmark, sync=ref_sync(sync_every_iter=True))
        timed(args, lazy_benchmark, sync=LazySync(sync_every_iter=True))
    warmup_time = time.perf_counter() - warmup0
    bench0 = time.perf_counter()
    for rep in range(args.repeat):
        # interleave the runs to handle frequency scaling and load changes
        # _, timings[rep, 0] = timed(args, benchmark, sync=ref_sync(sync_every_iter=True))
        _, timings[rep, 1] = timed(args, lazy_benchmark, sync=LazySync(skip_final_sync=True))
        xm.wait_device_ops()
        if current_device == 'cuda':
            torch.cuda.synchronize()
    lazy_metrics = dump_lazy_metrics()
    bench_time = time.perf_counter() - bench0
    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    median = np.median(timings, axis=0)
    fallbacks = ";".join([f"{m}:{lazy_metrics[m]}" for m in lazy_metrics if "aten::" in m])
    ops = int(sum([lazy_metrics[m] for m in lazy_metrics if 'xla::' in m or 'aten::' in m]) / args.repeat)
    trace_us = median[1] / 1e-6
    us_per_op = trace_us / ops
    overhead = median[1] / median[0]
    results.append(overhead)
    
    output_csv(
        os.path.join(args.output_dir, f"lazy-overheads_{args.test}_lala.csv"),
        # os.path.join(args.output_dir, f"lazy-overheads_{args.test}_{get_unique_suffix()}.csv"),
        ("dev", "name", "test", "overhead", "pvalue", "ops", "trace_us", "us_per_op", "fallbacks", "bench_time", "steps"),
    ).writerow([current_device, current_name, args.test, f"{overhead:.4f}", f"{pvalue:.4e}",
                f"{ops}", f"{trace_us:.4f}", f"{us_per_op:.4f}", f"{fallbacks}", f"{bench_time:.2f}", f"{args.repeat}"])
    
    print(f"{short_name(current_name, limit=30):<30} {current_device:<4} {args.test:<5} "
          f"{'trace overheads':<20} overhead: {overhead:.3f} pvalue: {pvalue:.2e} us_per_op {us_per_op:.3f}")
    if args.verbose:
        print(f"CIDEBUGOUTPUT,lazy_overhead_experiment,"
              f"{current_name},{args.test},{current_device},{overhead:.4f},"
              f"{pvalue:.4e},{args.warmup},{args.repeat},{warmup_time:.2f},{bench_time:.2f}")
    return (overhead, pvalue)


def lazy_overhead_experiment(args, results, benchmark, lazy_benchmark):
    timings = np.zeros((args.repeat, 2), np.float64)
    warmup0 = time.perf_counter()
    for rep in range(args.warmup):
        # interleave the runs to handle frequency scaling and load changes
        timed(args, benchmark, sync=ref_sync(sync_every_iter=True))
        timed(args, lazy_benchmark, sync=LazySync(sync_every_iter=True))
    warmup_time = time.perf_counter() - warmup0
    bench0 = time.perf_counter()
    for rep in range(args.repeat):
        # interleave the runs to handle frequency scaling and load changes
        _, timings[rep, 0] = timed(args, benchmark, sync=ref_sync(sync_every_iter=True))
        _, timings[rep, 1] = timed(args, lazy_benchmark, sync=LazySync(skip_final_sync=True))
        xm.wait_device_ops()
        if current_device == 'cuda':
            torch.cuda.synchronize()
    lazy_metrics = dump_lazy_metrics()
    bench_time = time.perf_counter() - bench0
    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    median = np.median(timings, axis=0)
    fallbacks = ";".join([f"{m}:{lazy_metrics[m]}" for m in lazy_metrics if "aten::" in m])
    ops = int(sum([lazy_metrics[m] for m in lazy_metrics if 'xla::' in m or 'aten::' in m]) / args.repeat)
    trace_us = median[1] / 1e-6
    us_per_op = trace_us / ops
    overhead = median[1] / median[0]
    results.append(overhead)

    output_csv(
        os.path.join(args.output_dir, f"lazy-overheads_{args.test}_lala.csv"),
        # os.path.join(args.output_dir, f"lazy-overheads_{args.test}_{get_unique_suffix()}.csv"),
        ("dev", "name", "test", "overhead", "pvalue", "ops", "trace_us", "us_per_op", "fallbacks", "bench_time", "step"),
    ).writerow([current_device, current_name, args.test, f"{overhead:.4f}", f"{pvalue:.4e}",
                f"{ops}", f"{trace_us:.4f}", f"{us_per_op:.4f}", f"{fallbacks}", f"{bench_time:.2f}", f"{args.repeat}"])
    print(f"{short_name(current_name, limit=30):<30} {current_device:<4} {args.test:<5} "
          f"{'trace overheads':<20} overhead: {overhead:.3f} pvalue: {pvalue:.2e} us_per_op {us_per_op:.3f}")
    if args.verbose:
        print(f"CIDEBUGOUTPUT,lazy_overhead_experiment,"
              f"{current_name},{args.test},{current_device},{overhead:.4f},"
              f"{pvalue:.4e},{args.warmup},{args.repeat},{warmup_time:.2f},{bench_time:.2f}")
    return (overhead, pvalue)

def lazy_compute_experiment(args, experiment, results, benchmark, lazy_benchmark, sync_every_iter=False):
    timings = np.zeros((args.repeat, 2), np.float64)
    lazy_sync = LazySync(sync_every_iter=sync_every_iter)

    # interleave the runs to handle frequency scaling and load changes
    warmup0 = time.perf_counter()
    for rep in range(args.warmup):
        # warmup
        timed(args, benchmark, sync=ref_sync)
        timed(args, lazy_benchmark, sync=lazy_sync)
    warmup_time = time.perf_counter() - warmup0

    # fresh metrics for each timed run
    dump_lazy_metrics(reset=True)
    bench0 = time.perf_counter()
    for rep in range(args.repeat):
        # measure
        _, timings[rep, 0] = timed(args, benchmark, times=args.inner_loop_repeat, sync=ref_sync)
        _, timings[rep, 1] = timed(args, lazy_benchmark, times=args.inner_loop_repeat, sync=lazy_sync)
    bench_time = time.perf_counter() - bench0
    lazy_metrics = dump_lazy_metrics(reset=True)
    if 'CachedCompile' not in lazy_metrics or lazy_metrics['CachedCompile'] != args.repeat * args.inner_loop_repeat:
        print("WARNING: lazy cached compile count indicates fallbacks, or something else")
    fallbacks = {k: v for (k, v) in lazy_metrics.items() if 'aten::' in k}
    if len(fallbacks):
        print(f"WARNING: lazy-eager fallbacks detected for [{fallbacks}]")
    if args.dump_lazy_counters:
        print(lazy_metrics)
    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    median = np.median(timings, axis=0)
    speedup = median[0] / median[1]
    results.append(speedup)
    output_csv(
        os.path.join(args.output_dir, f"lazy-compute_{args.test}_{get_unique_suffix()}.csv"),
        ("name", "dev", "experiment", "test", "speedup", "pvalue"),
    ).writerow([current_name, current_device, experiment, args.test, f"{speedup:.4f}", f"{pvalue:.2e}"])
    print(f"{short_name(current_name, limit=30):<30} {current_device:<4} "
          f"{args.test:<5} {experiment:<20} speedup:  {speedup:.3f} pvalue: {pvalue:.2e}")
    if args.verbose:
        print(f"CIDEBUGOUTPUT,lazy_compute_experiment,"
              f"{current_name},{current_device},{experiment},{args.test},{speedup:.4f},"
              f"{pvalue:.2e},{args.warmup},{args.repeat},{warmup_time:.2f},{bench_time:.2f}")
    return (speedup, pvalue)

def just_run_once(args, lazy_benchmark):
    set_seeds()
    if args.test == 'eval':
        model, example_inputs = lazy_benchmark.get_module()
        results.append(call_model_with(model, example_inputs))
    elif args.test == 'train':
        lazy_benchmark.train()
    xm.mark_step()
    xm.wait_device_ops()

def merge_with_prefix(prefix, tmp_dir, out_dir, headers):
    results = []
    rfnames = glob.glob(os.path.join(tmp_dir, prefix + "*"))
    for rfname in rfnames:
        results.extend(open(rfname).readlines()[1:])  # skip header

    # the header shouldn't require quotations and the results should already be properly
    # quoted via output_csv
    with open(os.path.join(out_dir, prefix + "acc.csv"), "a+") as acc_csv:
        acc_csv.write(",".join(headers) + "\n")
        for l in results:
            acc_csv.write(l)

def merge_reformat(tmp_dir, out_dir, table):
    out_dir = args.output_dir

    # depending on the type of an experiment, fields can be in a different order
    # `get_field` deals with all three types including `error`
    def get_field(row, name, file_type):
        headers = {
            "error": ("name", "test", "error"),
            "lazy-compute" : ("name", "dev", "experiment", "test", "speedup", "pvalue"),
            "lazy-overheads" : ("dev", "name", "test", "overhead", "pvalue", "ops", "trace_us", "us_per_op", "fallbacks")
        }

        header = headers[file_type]
        r = row[header.index(name)] if name in header else "N/A"
        return r

    csv_files = glob.glob(os.path.join(tmp_dir, "*.csv"))
    for csvf in csv_files:

        with open(csvf, "r") as csvfile:
            prefix = os.path.basename(csvf).split("_")[0]
            csvreader = csv.reader(csvfile, delimiter=",", quotechar='"')
            # This skips the first row of the CSV file.
            next(csvreader)

            for r in csvreader:
                key = (get_field(r, "name", prefix), get_field(r, "test", prefix))
                entry = table[key]

            if prefix == "error":
                entry["error"] = f'{entry.get("error", "")}  {get_field(r, "error", prefix)}'
            elif prefix == "lazy-overheads":
                entry["overhead"] = get_field(r, "overhead", prefix)
                entry["ops"] = get_field(r, "ops", prefix)
                entry["trace_us"] = get_field(r, "trace_us", prefix)
                entry["us_per_op"] = get_field(r, "us_per_op", prefix)
                entry["fallbacks"] = get_field(r, "fallbacks", prefix)
            else:
                entry[get_field(r, "experiment", prefix)] = get_field(r, "speedup", prefix)

    amortized_header = f"amortized {args.inner_loop_repeat}x"
    headers = ("name", "test", amortized_header, "unamortized", "overhead", "error", "rc",
               "ops", "trace_us", "us_per_op", "fallbacks")

    cw = output_csv(
        os.path.join(out_dir, f"{args.test}_reformat.csv"),
        headers
    )

    for k, v in table.items():
        cw.writerow((k[0], k[1], v.get(amortized_header, 'N/A'),
                     v.get('unamortized', 'N/A'), v.get('overhead', 'N/A'), v.get('error', 'N/A'), v.get('rc'),
                     v.get('ops', 'N/A'), v.get('trace_us', 'N/A'), v.get('us_per_op', 'N/A'), v.get('fallbacks', 'N/A')))

def save_error(name, test, error, dir):
    output_csv(
        os.path.join(dir, f"error_{get_unique_suffix()}.csv"),
        ("name", "test", "error"),
    ).writerow([name, test, error])


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", "-k", action="append", default=[], help="filter benchmarks")
    parser.add_argument("--exclude", "-x", action="append", default=[], help="filter benchmarks")
    parser.add_argument("--device", "-d", default='cuda', help="cpu or cuda")
    parser.add_argument("--warmup", type=int, default=4, help="number of warmup runs")
    parser.add_argument("--timeout", type=int, default=60 * 10, help="time allocated to each model")
    parser.add_argument("--repeat", "-n", type=int, default=4, help="number of timing runs (samples)")
    parser.add_argument("--inner_loop_repeat", type=int, default=10, help="repeat the computation this many times per sample")
    parser.add_argument("--test", type=str, choices=['eval', 'train'], default='eval')
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--torchbench_dir", type=str, help="path to torchbenchmark repo")
    parser.add_argument("--output_dir", type=str, default=".", help="path to write output files")
    parser.add_argument("--dump_lazy_counters", action='store_true', help="dump lazy counter values after each timing run")
    parser.add_argument("--just_run_once", action="store_true")
    parser.add_argument("--run_in_subprocess", "-s", type=str,
                        help="which model run in subprocess. This will ignore filter and exclude")
    parser.add_argument("--precision", choices=["fp32", "fp16", "amp"], default="fp32", help="enable fp16 modes from: fp32, fp16/half, or amp")
    args = parser.parse_args()
    results = []

    torchbench_dir = abspath(args.torchbench_dir) if args.torchbench_dir else abspath("../../benchmark")
    assert os.path.exists(os.path.join(torchbench_dir, "torchbenchmark")), "set --torchbench_dir to installed torchbench repo"
    sys.path.append(torchbench_dir)
    copy_argv = [] + sys.argv
    if args.run_in_subprocess:
        try:
            from fastNLP.core import logger
            logger.setLevel(logging.WARNING)
            current_name = args.run_in_subprocess
            benchmark_cls = get_benchmark_cls(args.run_in_subprocess)
            current_device = args.device
            if args.device == 'cuda':
                assert 'LTC_TS_CUDA' in os.environ and bool(os.environ['LTC_TS_CUDA']), "set LTC_TS_CUDA for cuda device"

            with pick_grad(args, current_name):

                    set_seeds()
                    lazy_benchmark = benchmark_cls(test=args.test, device=xm.xla_device(), jit=False, extra_args=["--precision", args.precision])
                    # TODO: might be redundant
                    gc.collect()

                    if args.just_run_once:
                        just_run_once(args, lazy_benchmark)
                        exit(0)

                    lazy_experiment(args, results, benchmark, lazy_benchmark)
                    # lazy_overhead_experiment(args, results, benchmark, lazy_benchmark)
                    #lazy_compute_experiment(args, f"amortized {args.inner_loop_repeat}x", results, benchmark, lazy_benchmark)
                    #lazy_compute_experiment(args, "unamortized", results, benchmark, lazy_benchmark, sync_every_iter=True)

        except Exception as e:
            print(f"ERROR: {current_name}: {e}")
            save_error(current_name, args.test, e, args.output_dir)
            exit(13)
        exit(0)

    import psutil
    import subprocess
    import tempfile
    dirpath = tempfile.mkdtemp()
    table = collections.defaultdict(dict)
    for model_name in iter_models(args, dirpath):
        print (model_name)
        # if `--run_in_subprocess` is specified, it will override any filters and excludes
        # pass the rest of arguments intact such as device, test, repeat, etc
        # note, the latest output_dir will override the original one and this is exactly what we want
        # for child processes
        launch_command = f"python {' '.join(copy_argv)} --run_in_subprocess '{model_name}' --output_dir={dirpath}"
        env = os.environ
        env["LTC_TS_CUDA"] = "1" if args.device == "cuda" else "0"
        rc = 0
        try:
            if args.verbose:
                cp = subprocess.run("nvidia-smi --query-gpu=timestamp,utilization.memory,memory.total,memory.free,memory.used"
                                    " --format=csv,noheader",
                                    capture_output=True, text=True, shell=True)
                print(f"CIDEBUGOUTPUT,BEFORE subprocess.run,{model_name},{cp.stdout}")
            proc = subprocess.Popen(launch_command,
                                    env=env,
                                    shell=True,
                                    stderr=subprocess.STDOUT)

            outs, errs = proc.communicate(timeout=args.timeout)
            rc = proc.poll()
        except subprocess.TimeoutExpired:
            print(f"{model_name} timed out after {args.timeout // 60} minutes! Include it in SKIP or SKIP_TRAIN_ONLY")
            save_error(model_name, args.test, "Timed out.", dirpath)
            # to visualize highlight timeouts, they will also have
            # "timed out" in the error column
            rc = 17
            process = psutil.Process(proc.pid)
            for p in process.children(recursive=True):
                p.kill()
            process.kill()
        if args.verbose:
            cp = subprocess.run("nvidia-smi --query-gpu=timestamp,utilization.memory,memory.total,memory.free,memory.used"
                                " --format=csv,noheader",
                                capture_output=True, text=True, shell=True)
            print(f"CIDEBUGOUTPUT,AFTER subprocess.run,{model_name},{args.test},{cp.stdout}")

        entry = table[(model_name, args.test)]
        entry["rc"] = rc
    merge_with_prefix("lazy-overheads_", dirpath, args.output_dir, ("dev", "name", "test", "overhead", "pvalue"))
    merge_with_prefix("lazy-compute_", dirpath, args.output_dir, ("name", "dev", "experiment", "test", "speedup", "pvalue"))
    merge_with_prefix("error_", dirpath, args.output_dir, ("name", "test", "error"))
    merge_reformat(dirpath, args, table)