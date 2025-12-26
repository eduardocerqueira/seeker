#date: 2025-12-26T17:11:55Z
#url: https://api.github.com/gists/965e072f33e744adde22743840ec2339
#owner: https://api.github.com/users/strnan

# SN38 Mechanism 1 — A6000-Optimized DiLoCo + SparseLoCo Strategy
# Designed for: low bandwidth, high throughput, stable loss
# Compatible with evaluator (4 nodes, 100 steps)

import math, os, datetime
import torch
import torch.distributed as dist
import torch.nn.utils as nn_utils
from dataclasses import dataclass
from copy import deepcopy
from typing import Optional, Union, Dict, Any, Iterable, Tuple, Type
from abc import ABC, abstractmethod

from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange

# =====================
# OPTIM SPEC
# =====================
@dataclass
class OptimSpec:
    cls: Type[torch.optim.Optimizer]
    kwargs: Dict[str, Any]

    def build(self, model):
        return self.cls(model.parameters(), **self.kwargs)

# =====================
# BASE STRATEGY
# =====================
class Strategy(ABC):
    def _init_node(self, model, rank, world):
        self.model = model
        self.rank = rank
        self.world = world
        self.local_step = 0
        self.optim = self.optim_spec.build(model)
        self._setup_scheduler()

    def _setup_scheduler(self):
        def lr_lambda(step):
            warmup = 10
            if step < warmup:
                return step / warmup
            return 0.5 * (1 + math.cos(math.pi * step / 100))
        self.scheduler = LambdaLR(self.optim, lr_lambda)

    @abstractmethod
    def step(self): ...

# =====================
# COMMUNICATION MODULE
# =====================
class CommunicationModule(ABC):
    @abstractmethod
    def communicate(self, model, rank, world, step): ...

# =====================
# DILOCO COMMUNICATOR
# =====================
class DiLoCoCommunicator(CommunicationModule):
    def __init__(self, H=15):
        self.H = H
        self.master = None

    def init(self, model):
        self.master = deepcopy(model).cpu()
        for p in self.master.parameters():
            p.requires_grad_(True)
        self.outer_optim = torch.optim.SGD(self.master.parameters(), lr=0.8, momentum=0.9)

    def communicate(self, model, rank, world, step):
        if world == 1 or step % self.H != 0 or step == 0:
            return
        if self.master is None:
            self.init(model)
        self.outer_optim.zero_grad()
        for (n, p), (_, mp) in zip(model.named_parameters(), self.master.named_parameters()):
            mp.grad = mp.data - p.data.cpu()
        self.outer_optim.step()
        for p, mp in zip(model.parameters(), self.master.parameters()):
            p.data.copy_(mp.data.to(p.device))

# =====================
# STRATEGY IMPLEMENTATION
# =====================
class DiLoCoSparseStrategy(Strategy):
    def __init__(self):
        self.optim_spec = OptimSpec(torch.optim.AdamW, dict(lr=1e-3))
        self.comm = DiLoCoCommunicator(H=15)
        self.max_norm = 1.0

    def step(self):
        nn_utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optim.step()
        self.comm.communicate(self.model, self.rank, self.world, self.local_step)
        self.scheduler.step()
        self.local_step += 1

# =====================
# ENTRYPOINT VARIABLE
# =====================
STRATEGY = DiLoCoSparseStrategy()