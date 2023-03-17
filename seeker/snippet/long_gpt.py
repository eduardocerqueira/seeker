#date: 2023-03-17T17:04:13Z
#url: https://api.github.com/gists/1c36eaddd03ed102d24372493264694c
#owner: https://api.github.com/users/NaxAlpha

import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

import wandb
from tqdm import tqdm

from datasets import load_dataset
from transformers import GPT2TokenizerFast

# copy from here: https://github.com/karpathy/nanoGPT/blob/master/model.py
from ngpt import GPT

WANDB_STYLE = """
<style>
    html, body {
        padding: 0;
        margin: 0;
        width: 100%;
        height: 100%;
    }
    p {
        font-family: 'Verdana', sans-serif;
    }
</style>
"""


def closest_power_of_2(x):
    return 2 ** (x - 1).bit_length()


class DatasetWrapper(IterableDataset):
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"_ "**********"i "**********"n "**********"i "**********"t "**********"_ "**********"_ "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"m "**********"a "**********"x "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"= "**********"2 "**********"* "**********"* "**********"1 "**********"2 "**********") "**********": "**********"
        self.ds = load_dataset(
            "the_pile",
            name="all",
            split="train",
            streaming=True,
        ).shuffle(buffer_size=100_000)
        self.tokenizer = "**********"
        self.max_tokens = "**********"

    def __iter__(self):
        buffer = []
        for sample in self.ds:
            buffer += "**********"
            buffer += "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"w "**********"h "**********"i "**********"l "**********"e "**********"  "**********"l "**********"e "**********"n "**********"( "**********"b "**********"u "**********"f "**********"f "**********"e "**********"r "**********") "**********"  "**********"> "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"m "**********"a "**********"x "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********": "**********"
                yield torch.tensor(buffer[: "**********"
                buffer = buffer[self.max_tokens : "**********"


class Trainer:
    def __init__(self):
        self.tokenizer: "**********"
        self.max_tokens = "**********"
        self.grad = 1
        self.step = 0

        self.dataset = "**********"
        self.loader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=8,
        )
        self.model = model = GPT.from_pretrained("gpt2").cuda()
        # model.load_state_dict(torch.load("v2.pt"))

        self.opt = model.configure_optimizers(
            weight_decay=1e-1,
            learning_rate=1e-6,
            betas=(0.9, 0.95),
            device_type="cuda",
        )

        # patch model embeddings
        emb = model.transformer.wpe.weight.data
        wpe = "**********"
        wpe.weight.data = "**********"
        model.transformer.wpe = wpe
        model.config.block_size = "**********"
        print("Patched model embeddings:", wpe.weight.shape)

        self.model = torch.compile(model)

    def train_step(self, batch):
        batch = batch.cuda()
        x, y = batch[:, :-1], batch[:, 1:].contiguous()
        _, loss = self.model(x, targets=y)
        (loss / self.grad).backward()
        return loss

    def generate_samples(self, n_samples=8):
        x = "**********"
        t0 = time.time()
        self.model.eval()
        y = "**********"=1100).tolist()
        self.model.train()
        t1 = time.time()
        t = "**********"
        t = "<hr>".join(f"<p>{c}</p>" for c in t)
        html = WANDB_STYLE + t
        wandb.log({"samples": wandb.Html(html)}, step=self.step)
        print(f"Generated in {t1-t0:.3f}s")

    def train(self):
        wandb.init(
            project="long-gpt",
            entity="...",
        )

        prog = tqdm(self.loader)
        self.opt.zero_grad()

        for i, batch in enumerate(prog):
            self.step = i + 1

            loss = self.train_step(batch)
            prog.set_description(f"loss: {loss.item():.3f}")
            wandb.log({"loss": loss.item(), "grad": self.grad}, step=i)

            if i % self.grad == 0:
                self.opt.step()
                self.opt.zero_grad()
            self.grad = closest_power_of_2(i + 1)

            if i % 100 == 0:
                torch.save(self.model.state_dict(), "model.pt")

            if i % 1000 == 0:
                self.generate_samples(8)


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
