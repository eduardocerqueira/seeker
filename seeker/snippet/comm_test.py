#date: 2025-04-14T16:58:04Z
#url: https://api.github.com/gists/2bc71c14bef0a448915f915293757db3
#owner: https://api.github.com/users/HollowMan6

import torch
import torch.distributed as dist
from datetime import timedelta

import ray
import os
from abc import ABC
from collections import defaultdict
from typing import Callable, List, T

import ray
from ray.util.placement_group import placement_group, PlacementGroupSchedulingStrategy
from ray.actor import ActorHandle
from ray.train._internal.utils import get_address_and_port


class TorchDistributedWorker(ABC):
    """Defines the interfaces required by the init_torch_dist_process_group().

    This is modeled after RayTrainerWorker, which allows arbitrary functions
    to be executed on a remote DDP worker.
    """

    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Executes the input function and returns the output.

        Args:
            func: The function to execute.
            args, kwargs: The arguments to pass into func.
        """
        return func(*args, **kwargs)


def _init_torch_distributed(
    init_method: str,
    backend: str,
    rank: int,
    world_size: int,
    local_rank: int,
    local_world_size: int,
    master_addr: str,
    master_port: str,
    gpu_ids: List[int],
    **init_process_group_kwargs,
):
    """Initialize torch distributed backend"""
    if init_method == "env":
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)
        url = "env://"
    elif init_method == "tcp":
        url = f"tcp://{master_addr}:{master_port}"
    else:
        raise ValueError(
            f"The provided init_method ("
            f"{init_method}) is not supported. Must "
            f"be either 'env' or 'tcp'."
        )
    
    original_gpu_id = gpu_ids[local_rank]
    gpu_ids = [original_gpu_id]
    local_rank = 0
    local_world_size = 1
    torch.cuda.set_device(f"cuda:0") #{local_rank}")

    if backend == "nccl":
        # Same as in Ray Train
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # All workers on a same node should share the same set of
        # visible GPUs. Otherwise they can't talk among themselves.
        device_ids = ",".join(map(str, gpu_ids))
        print(f"====={os.environ['CUDA_VISIBLE_DEVICES']}====")
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
        # print(f"====={os.environ['ROCR_VISIBLE_DEVICES']}====")
        # os.environ["ROCR_VISIBLE_DEVICES"] = device_ids
        # os.environ["HIP_VISIBLE_DEVICES"] = device_ids
        print(f"****{os.environ['CUDA_VISIBLE_DEVICES']}****")

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)

    print(
        f"RANK: {rank}, LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}, LOCAL_WORLD_SIZE: {local_world_size}"
    )

    dist.init_process_group(backend, timeout=timedelta(seconds=120), init_method=url) #, device_id=torch.device(f"cuda:{local_rank}"))


def _get_node_and_gpu_ids():
    """Returns the node_id and gpu_ids for this worker."""
    node_id = ray.get_runtime_context().get_node_id()
    gpu_ids = ray.get_gpu_ids()
    return node_id, gpu_ids


def init_torch_dist_process_group(
    workers: List[ActorHandle],
    backend: str = "gloo",
    init_method: str = "env",
    **init_process_group_kwargs,
) -> List[int]:
    """Initialize a torch distributed process group.

    Note: this util assumes that the order of the workers passed in
    are their global ranks.

    Args:
        workers: A list of TorchDistributedWorker actors.
        backend: The torch distributed backend to use,
            possible choices are "gloo" or "nccl".
        init_method: The initialization method to use,
            possible choices are "env" or "tcp".
        init_process_group_kwargs: Additional kwargs to pass to the call to
            :meth:`torch.distributed.init_process_group`.

    Returns:
        Local ranks on their respective nodes for the list of workers.
    """
    if not dist.is_available():
        raise RuntimeError("Distributed torch is not available.")

    # Build a map from node_id to workers on that node.
    node_and_gpu_ids = ray.get(
        [w.execute.remote(_get_node_and_gpu_ids) for w in workers]
    )
    # All the workers on a specific node.
    node_to_workers = defaultdict(list)
    # All the gpu ids visible to all the workers on a specific node.
    node_to_gpu_ids = defaultdict(set)
    for i, (node_id, gpu_ids) in enumerate(node_and_gpu_ids):
        node_to_workers[node_id].append(i)
        # Force list.
        if not isinstance(gpu_ids, list):
            gpu_ids = [gpu_ids]
        # It is possible for a worker to have access to multiple GPUs.
        for gpu_id in gpu_ids:
            node_to_gpu_ids[node_id].add(gpu_id)

    # Assume the first worker is the master.
    master_addr, master_port = ray.get(workers[0].execute.remote(get_address_and_port))

    setup_futures = []
    world_size = len(workers)
    local_ranks = []
    for rank, worker in enumerate(workers):
        node_id = node_and_gpu_ids[rank][0]
        local_rank = node_to_workers[node_id].index(rank)
        local_world_size = len(node_to_workers[node_id])
        setup_futures.append(
            worker.execute.remote(
                _init_torch_distributed,
                init_method=init_method,
                backend=backend,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
                local_world_size=local_world_size,
                master_addr=master_addr,
                master_port=master_port,
                gpu_ids=list(node_to_gpu_ids[node_id]),
                **init_process_group_kwargs,
            )
        )
        local_ranks.append(local_rank)

    # Wait for all workers to join the process group.
    ray.get(setup_futures)

    return local_ranks


def _shutdown_torch_distributed():
    """Shutdown torch distributed backend"""
    dist.destroy_process_group()

    if not torch.cuda.is_available():
        return


def shutdown_torch_dist_process_group(workers: List[ActorHandle]):
    ray.get([w.execute.remote(_shutdown_torch_distributed) for w in workers])


def test_torch_process_group(num_workers: int, backend: str):
    @ray.remote(num_gpus=1, num_cpus=1)
    class TestWorker(TorchDistributedWorker):
        def __init__(self):
            super().__init__()
            self.dev = f"cuda:{ray.get_gpu_ids()[0]}"
            print(f"device: {self.dev}")
            print(f"device_count: {torch.cuda.device_count()}")

        def run(self):
            line = ""
            for key in [
                "CDUA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "LOCAL_WORLD_SIZE",
            ]:
                line += f"{key} {os.environ.get(key)}, "

            print(line)

            # with open(f'ray-{os.environ["RANK"]}.json', 'w') as f:
            #     import json
            #     json.dump(dict(os.environ), f)

            self.dev = f"cuda:{os.environ['LOCAL_RANK']}"

            print("Start running, waiting for barrier...")
            dist.barrier()
            print("Barrier passed.")
            tensor = torch.tensor([1.0]).to(self.dev)

            print("Running allreduce")
            dist.all_reduce(tensor)
            print("Allreduce finished..")

            print("Running broadcast")
            # test broadcast
            if dist.get_rank() == 0:
                self.send = torch.ones(
                    (4,), dtype=torch.float32, device=self.dev
                )
            else:
                self.send = torch.zeros(
                    (4,), dtype=torch.float32, device=self.dev
                )

            print(f"Rank {dist.get_rank()} computing broadcast")
            dist.broadcast(self.send, 0)
            print(f"Rank {dist.get_rank()} done computing broadcast")
            print(self.send)
            print("Broadcast finished.")

            # import time
            # time.sleep(30)
            # broadcast_group = [0, 12]
            # new_group = dist.new_group(broadcast_group, backend=backend, use_local_synchronization=True)
            # print(f"New group created: {new_group}")

            # if dist.get_rank() in broadcast_group:
            #     source_rank = broadcast_group[-1]
            #     if dist.get_rank() == source_rank:
            #         tensor = torch.tensor([123.0]).to(self.dev)
            #         print(f"Rank {source_rank} broadcasting tensor: {tensor}")
            #     else:
            #         tensor = torch.tensor([0.0]).to(self.dev)
            #         print(f"Rank {dist.get_rank()} before broadcast: {tensor}")

            #     dist.broadcast(tensor, source_rank, group=new_group)
            #     print(f"Broadcast passed for new group")

            return tensor.cpu().numpy()

    pg = placement_group([{"CPU": 1, "GPU": 1}] * num_workers, strategy="PACK")
    workers = [
        TestWorker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
            )
        ).remote()
        for _ in range(num_workers)
    ]

    local_ranks = init_torch_dist_process_group(
        workers, backend=backend, init_method="env"
    )
    print(f"local_ranks: {local_ranks}")

    reduced = ray.get([w.run.remote() for w in workers])

    print(f"All test passed. Results: {reduced}")

    shutdown_torch_dist_process_group(workers)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", "--num_workers", type=int, default=4)
    args = parser.parse_args()
    num_workers = args.num_workers
    runtime_env = {
        "env_vars": {
            # "NCCL_DEBUG": "TRACE",
            # "NCCL_DEBUG_SUBSYS": "ALL,COLL",
            # "LOG_LEVEL": "debug",
            "NCCL_NET_GDR_LEVEL": "3",
            # "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
            # "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        }
    }
    ray.init(address="auto", runtime_env=runtime_env)
    # test_torch_process_group(num_workers, "nccl")
    test_torch_process_group(num_workers, "gloo")
