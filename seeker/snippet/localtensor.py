#date: 2025-11-20T16:55:37Z
#url: https://api.github.com/gists/9ef3300cd516150637592af1e8b76d7d
#owner: https://api.github.com/users/ezyang

import torch
from torch import Tensor
from torch.distributed.tensor import (
    DTensor,
    DeviceMesh,
    distribute_tensor,
    init_device_mesh,
    Partial,
    Replicate,
    Shard,
)
from torch.distributed.tensor.placement_types import _StridedShard
from torch.distributed._local_tensor import (
    local_tensor_mode,
    LocalTensor,
    LocalTensorMode,
)
import traceback

S = Shard
R = Replicate()
_SS = _StridedShard

def product(it):
    x = 1
    for i in it:
        x *= i
    return x

def arange_nd(*sizes):
    if len(sizes) == 1 and isinstance(sizes[0], (list, tuple)):
        sizes = sizes[0]
    return torch.arange(product(sizes)).view(sizes)

def reconcile(l: Tensor):
    """Asserts that a LocalTensor is the same on all ranks, and returns the single Tensor."""
    if isinstance(l, LocalTensor):
        return l.reconcile()
    return l

def init_local_tensor_mode(world_size):
    from torch.distributed import _local_tensor
    if _local_tensor._LOCAL_TENSOR_MODE:
        for lm in list(reversed(_local_tensor._LOCAL_TENSOR_MODE)):
            lm.__exit__(None, None, None)
    try:
        torch.distributed.destroy_process_group()
    except AssertionError:
        pass
    torch.distributed.init_process_group(
        "fake",
        rank=0,
        world_size=world_size,
    )
    lm = LocalTensorMode(world_size)
    lm.__enter__()
    return world_size

def init_fake_tensor_mode(world_size):
    from torch.distributed import _local_tensor
    if _local_tensor._LOCAL_TENSOR_MODE:
        for lm in list(reversed(_local_tensor._LOCAL_TENSOR_MODE)):
            lm.__exit__(None, None, None)
    try:
        torch.distributed.destroy_process_group()
    except AssertionError:
        pass
    torch.distributed.init_process_group(
        "fake",
        rank=0,
        world_size=world_size,
    )
    return world_size
    
world_size = init_local_tensor_mode(4)
mesh = init_device_mesh("cpu", (4,), mesh_dim_names=("x",))
a = DTensor.from_local(arange_nd(4).float(), mesh, [R])
b = DTensor.from_local(torch.ones(4), mesh, [Partial()])
a += b
print(a)