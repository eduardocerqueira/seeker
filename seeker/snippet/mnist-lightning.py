#date: 2022-05-06T17:06:40Z
#url: https://api.github.com/gists/33773389cf95be8d187f6e217c9031e0
#owner: https://api.github.com/users/jchia

#!/bin/env python3

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CyclicLR

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.cli import LightningCLI

from typing import Optional


import pdb

class MnistDataModule(LightningDataModule):

    def __init__(self, batch_size: int=64, test_batch_size: int=1000):
        """
        Args:
            batch_size: training batch size
            test_batch_size: testing batch size
        """
        super().__init__()
        self._batch_size = batch_size
        self._test_batch_size = test_batch_size

    def prepare_data(self):
        print('PREPARE-DATA')
        datasets.MNIST('data', train=True, download=True)
        datasets.MNIST('data', train=False)

    def setup(self, stage: Optional[TrainerFn] = None):
        print('SETUP', stage)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        kwargs = {'num_workers': 1, 'pin_memory': True}

        if stage == TrainerFn.FITTING or stage is None:
            kwargs['batch_size'] = self._batch_size
            kwargs['shuffle'] = True
            self._train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        if stage == TrainerFn.TESTING or stage is None:
            kwargs['batch_size'] = self._test_batch_size
            kwargs['shuffle'] = False
            self._test_dataset = datasets.MNIST('data', train=False, transform=transform)

    def train_dataloader(self):
        kwargs = {'batch_size': self._batch_size,
                  'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_loader = torch.utils.data.DataLoader(self._train_dataset, **kwargs)
        return train_loader

    def test_dataloader(self):
        kwargs = {'batch_size': self._test_batch_size,
                  'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        test_loader = torch.utils.data.DataLoader(self._test_dataset, **kwargs)
        return test_loader


class MnistModule(LightningModule):
    def __init__(self, lr: float):
        """Test MNIST model

        Args:
            lr: min learning rate
        """
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(3, 1, 28, 28)
        self.lr = lr
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.nll_loss(self(x), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        num_samples = y.size(0)
        output = self(x)
        loss = F.nll_loss(output, y, reduction='sum')
        pred = output.argmax(dim=1, keepdim=False)  # get the index of the max log-probability
        correct = pred.eq(y).sum()
        self.log_dict({'val_loss_sum1': loss,
                       'val_loss1': loss / num_samples,
                       'val_accuracy1': correct / num_samples},
                      prog_bar=True)
        return {'n': num_samples, 'l': loss, 'c': correct}

    def validation_epoch_end(self, outputs) -> None:
        n = sum(x['n'] for x in outputs)
        l = sum(x['l'] for x in outputs)
        c = sum(x['c'] for x in outputs)
        loss = l / n
        accuracy = c / n
        self.log_dict({'val_loss2': loss,
                       'val_accuracy2': accuracy,
                       'val_n': float(n)})

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs) -> None:
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        scheduler = CyclicLR(optimizer, base_lr=self.lr, max_lr=self.lr * 10)
        return [optimizer], [scheduler]


def cli_main():
    cli = LightningCLI(
        MnistModule,
        MnistDataModule,
        seed_everything_default=1234,
        save_config_overwrite=True,
        run=False,  # Disable automatic fitting.
        trainer_defaults={'max_epochs': 3, 'accelerator': 'gpu', 'devices': 1},
    )

    print('HPARAMS', cli.model.hparams)

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, datamodule=cli.datamodule)

if __name__ == '__main__':
    # main()
    cli_main()
