#date: 2025-10-09T16:57:16Z
#url: https://api.github.com/gists/95cecc34b742cc1d4fdaac9fb7d8da26
#owner: https://api.github.com/users/epignatelli

import os
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm


@dataclass
class Args:
    num_iter: int = 100
    log_every: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Classifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor], optimiser: torch.optim.Optimizer) -> Dict[str, Any]:
        # unpack
        xb, yb = batch

        # loss
        logits = self(xb)
        loss = F.mse_loss(logits, yb.float())

        # backprop
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        # log
        acc = logits.argmax(dim=1).eq(yb.argmax(dim=1)).float().mean()
        return {"train/loss": loss.item(), "train/acc": acc.item()}

    def sample_data(self, dataset: datasets.MNIST, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = random.choices(dataset.data, k=batch_size)
        ys = random.choices(dataset.targets, k=batch_size)
        xs, ys = torch.stack(xs).view(batch_size, -1).to(device), torch.stack(ys).to(device)
        ys = torch.nn.functional.one_hot(ys, num_classes=10)
        return xs.float(), ys.long()

    def optimise(self, args: Args) -> List[Dict[str, Any]]:
        # seed
        seed_everything(args.seed)
        logger = logging.getLogger("train")
        logger.setLevel(logging.INFO)

        # data
        mnist = datasets.FashionMNIST("data", download=True, train=True)

        # optimiser
        optimiser = torch.optim.Adam(self.parameters(), lr=args.lr)

        # train
        history = []
        for i in range(args.num_iter):
            xs, ys = self.sample_data(mnist, args.batch_size, args.device)
            logs = self.train_step((xs, ys), optimiser)
            logs["train/iter"] = i
            history.append(logs)

            if i % args.log_every == 0:
                logger.info(logs)

        return history

    def plot(self, history: List[Dict[str, Any]]) -> None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        for k, v in history[0].items():
            if "train/acc" in k:
                plt.plot([log[k] for log in history], label=k)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    args = Args(
        num_iter=100,
        log_every=10,
        batch_size=4,
        lr=1e-3,
        seed=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = Classifier().to(args.device)
    logs = model.optimise(args)
    model.plot(logs)