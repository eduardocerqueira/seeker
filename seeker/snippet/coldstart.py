#date: 2022-06-08T16:47:45Z
#url: https://api.github.com/gists/2605f8c3c3b5bb7a13afb8e63eae9553
#owner: https://api.github.com/users/wconstab

import argparse
import torch
parser = argparse.ArgumentParser()
parser.add_argument("--dynamo", action="store_true")
parser.add_argument("--size", type=int,  default=1)
parser.add_argument("--child", action="store_true", help="inside child process")
parser.add_argument("--repeat", type=int, default=2, help="how many repeats (without warmup) to time. 2 covers profiling executor behavior.")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()


def func(x):
    return torch.sum(torch.relu(x + 1))

if args.child:
    x = torch.randn((args.size, ), device=args.device)
    for i in range(args.repeat):
        if args.dynamo:
            import torchdynamo
            from torchdynamo.optimizations.training import aot_autograd_speedup_strategy
            with torchdynamo.optimize(aot_autograd_speedup_strategy):
                func(x)
        else:
            func(x)
else:
    import subprocess
    import sys
    import time

    def timed_(size, dynamo=False):
        experiment_args = [
            f"--size {size}",
            "--dynamo" if dynamo else ""
        ]
        launch_command = f"python {' '.join(copy_argv)} --child {' '.join(experiment_args)}"
        
        t0 = time.time()
        proc = subprocess.Popen(launch_command,
                                shell=True,
                                stderr=subprocess.STDOUT)
        outs, errs = proc.communicate()
        rc = proc.poll()
        assert rc == 0
        t = time.time() - t0
        return t

    copy_argv = [] + sys.argv
    for size in [1, 128, 1024, 2**16, 2**20]:
        t_eager = timed_(size) 
        t_dynamo = timed_(size, dynamo=True)
        speedup = t_eager / t_dynamo
        print(f"size {size} eager: {t_eager} dynamo: {t_dynamo} {speedup}")