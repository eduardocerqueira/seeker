#date: 2021-12-31T16:27:07Z
#url: https://api.github.com/gists/54dac56efec56086f124dad0535b5ce4
#owner: https://api.github.com/users/crowsonkb

#!/usr/bin/env python3

"""Learns the parity function."""

import torch
from torch import nn, optim
from tqdm import trange, tqdm


class GatedUnit(nn.Module):
    def __init__(self, act=None):
        super().__init__()
        self.act = act if act else nn.Identity()

    def forward(self, input):
        a, b = input.chunk(2, dim=1)
        return a * self.act(b)


class ParityBlock(nn.Module):
    def __init__(self, d_in, d_out, d_model, d_ff):
        super().__init__()
        self.to_hidden = nn.Sequential(
            nn.LayerNorm(d_model + d_in),
            nn.Linear(d_model + d_in, d_ff * 2),
            GatedUnit(nn.LeakyReLU(0.2)),
        )
        self.to_next = nn.Linear(d_ff, d_model)
        self.to_out = nn.Linear(d_ff, d_out)

    def forward(self, res_input, seq_input, scale_by):
        input = torch.cat([res_input, seq_input], dim=1)
        hidden = self.to_hidden(input)
        return res_input + self.to_next(hidden) * scale_by, self.to_out(hidden)


class ParityModel(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.block = ParityBlock(1, 1, d_model, d_ff)

    def forward(self, input):
        x = input.new_zeros([input.shape[0], self.d_model])
        scale_by = 1 / input.shape[1]
        for i in range(0, input.shape[1]):
            x, out = self.block(x, input[:, i:i+1], scale_by)
        return out


def sample_data(n, bits, device='cpu'):
    inputs = torch.randint(2, [n, bits], device=device).float()
    targets = inputs.sum(1, keepdim=True) % 2
    return inputs, targets


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    bits = 30
    steps = 30000
    batch_size = 2**15

    model = ParityModel(d_model=24, d_ff=64).to(device)
    print('Parameters:', sum(p.numel() for p in model.parameters()))

    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3, betas=(0.95, 0.95))
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    for i in trange(steps):
        inputs_batch, targets_batch = sample_data(batch_size, bits, device=device)
        opt.zero_grad()
        outputs = model(inputs_batch * 2 - 1)
        loss = loss_fn(outputs, targets_batch)
        accuracy = ((outputs > 0) == targets_batch).sum(0) / batch_size
        loss.backward()
        opt.step()
        sched.step()
        tqdm.write(f'step: {i}, loss: {loss.item():g}, acc: {accuracy.item():g}')


if __name__ == '__main__':
    main()
