#date: 2025-11-20T17:13:07Z
#url: https://api.github.com/gists/ab4d56b1680932bd8a00b7d658e88ba3
#owner: https://api.github.com/users/river

"""
Test pytorch multi gpu runs

Usage (from repo root, with uv):

  uv run scripts/ddp_sanity_check.py --devices 0 1

This will:
  - spawn one process per device
  - run a few training steps on random data
  - print success / any obvious errors per rank
"""

import argparse
import os
from typing import List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim


class ToyModel(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def ddp_worker(rank: int, world_size: int, device_ids: List[int], steps: int, backend: str) -> None:
    device_index = device_ids[rank]
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")

    # Each process needs a unique rank and the same world_size
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    torch.cuda.set_device(device_index) if device.type == "cuda" else None

    model = ToyModel().to(device)
    ddp_model = nn.parallel.DistributedDataParallel(
        model, device_ids=[device_index] if device.type == "cuda" else None
    )

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    if rank == 0:
        print(f"[rank {rank}] Starting DDP sanity run on devices {device_ids}")

    for step in range(steps):
        # Fake data
        x = torch.randn(32, 128, device=device)
        y = torch.randn(32, 128, device=device)

        optimizer.zero_grad()
        y_pred = ddp_model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        if step == steps - 1:
            print(f"[rank {rank}] Finished step {step + 1}/{steps}, loss={loss.item():.4f}")

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny DDP sanity check")
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        required=True,
        help="List of CUDA device indices to use, e.g. --devices 0 1",
    )
    parser.add_argument(
        "--steps", type=int, default=5, help="Number of training steps to run per process"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend to use (nccl for GPUs, gloo for CPU)",
    )
    args = parser.parse_args()

    devices = args.devices
    world_size = len(devices)
    if world_size < 2:
        raise ValueError(
            "Provide at least two devices for a multi-GPU DDP test, e.g. --devices 0 1"
        )

    # Required for init_method='env://'
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    print(f"Spawning {world_size} processes for devices: {devices} with backend={args.backend}")

    mp.spawn(
        ddp_worker,
        args=(world_size, devices, args.steps, args.backend),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()