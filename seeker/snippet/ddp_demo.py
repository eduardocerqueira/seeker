#date: 2026-03-05T18:27:45Z
#url: https://api.github.com/gists/09b4fa28456aba24cab8efa2432da600
#owner: https://api.github.com/users/r10a

"""

echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt update
apt-get install -y gnupg
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install -y nsight-systems-cli
nsys --version && nsys status -e

nsys profile \
  --sample process-tree \
  --trace cuda,nvtx,osrt,cudnn,cublas,mpi \
  --cuda-graph-trace node \
  --cudabacktrace=kernel \
  --trace-fork-before-exec true \
  --gpu-metrics-devices all \
  --python-sampling=true \
  --cuda-memory-usage true \
  --stats=true \
  --output /tmp/tritonserver.nsys-rep \
  --force-overwrite true \
  python3 ddp_demo.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import os

class MultiLayerNet(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    model = MultiLayerNet(1024).to(rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    #Each rank makes its own data
    batch_size = 256
    data = torch.randn(batch_size, 1024, device=rank)
    target = torch.randn(batch_size, 1, device=rank)
    output = ddp_model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward() #DDP overlaps gradient all-reduce on background stream
    optimizer.step()
    dist.destroy_process_group()

def train_no_overlap(rank, world_size):  
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", world_size=world_size, rank=rank)  
    torch.cuda.set_device(rank) 
    #Each rank synthesizes its own data  #(avoid sending big tensors via spawn)  
    batch_size = 256  
    data = torch.randn(batch_size, 1024, device=rank)  
    target = torch.randn(batch_size, 1, device=rank)  
    model = MultiLayerNet(data.size(1)).to(rank)  
    optimizer = optim.SGD(model.parameters(), lr=0.01)  
    #Forward + backward (manual, no overlap)  
    output = model(data)  
    loss = nn.functional.mse_loss(output, target)  
    loss.backward()  
    #Synchronous gradient all-reduce after backward  
    for p in model.parameters():  
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)  
        p.grad /= world_size  
    
    optimizer.step()  
    dist.destroy_process_group()  

def main():
    world_size = min(2, torch.cuda.device_count() or 1)
    mp.set_start_method("spawn", force=True)
    if world_size > 1:
        print("train_no_overlap")
        mp.spawn(train_no_overlap, args=(world_size,),  nprocs=world_size,  join=True)
        print("train_ddp")
        mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("Only one GPU present; running DDP demo with world_size=1")
        print("train_no_overlap")
        train_no_overlap(0, 1)
        print("train_ddp")
        train_ddp(0, 1)

if __name__ == "__main__":
    main() 