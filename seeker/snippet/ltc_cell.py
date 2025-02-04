#date: 2025-02-04T16:53:00Z
#url: https://api.github.com/gists/04fcb82e868d191062d76ab8abeaf0b0
#owner: https://api.github.com/users/Achronus

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTCCell(nn.Module):
    """
    A Liquid Time-Constant (LTC) cell following the closed-form continuous-depth
    (CFC; Equation 10) solution from the paper: https://arxiv.org/abs/2106.13898.

    Equation:
    x(t) =
        σ(-f(x, I, θ_f), t) g(x, I, θ_g)
        + [1 - σ(-[f(x, I, θ_f)]t)] h(x, I, θ_h)

    Parameters:
        in_features (int): number of input nodes
        n_hidden (int): number of hidden nodes
        backbone (nn.Module): a custom Neural Network backbone. E.g., an MLP or ConvNet
    """

    def __init__(
        self,
        in_features: int,
        n_hidden: int,
        backbone: nn.Module,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.n_hidden = n_hidden
        self.backbone = backbone

        self.tanh = nn.Tanh()  # Bounded: [-1, 1]
        self.sigmoid = nn.Sigmoid()  # Bounded: [0, 1]

        head_size = n_hidden + in_features
        self.g_head = nn.Linear(head_size, n_hidden)
        self.h_head = nn.Linear(head_size, n_hidden)

        # LTC heads (f)
        self.f_head_to_g = nn.Linear(head_size, n_hidden)
        self.f_head_to_h = nn.Linear(head_size, n_hidden)

    def _new_hidden(
        self,
        x: torch.Tensor,
        g_out: torch.Tensor,
        h_out: torch.Tensor,
        ts: int,
    ) -> torch.Tensor:
        """
        Computes the new hidden state.

        Parameters:
            x (torch.Tensor): input values
            g_out (torch.Tensor): g_head output
            h_out (torch.Tensor): h_head output
            ts (torch.Tensor): current hidden timestep

        Returns:
            hidden (torch.Tensor): a new hidden state
        """
        g_head = self.tanh(g_out)  # g(x, I, θ_g)
        h_head = self.tanh(h_out)  # h(x, I, θ_h)

        fh_g = self.f_head_to_g(x)
        fh_h = self.f_head_to_h(x)

        gate_out = self.sigmoid(fh_g * ts + fh_h)  # [1 - σ(-[f(x, I, θf)], t)]
        f_head = 1.0 - gate_out  # σ(-f(x, I, θf), t)

        return g_head * f_head + gate_out * h_head

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, ts: int) -> torch.Tensor:
        """
        Performs a forward pass through the cell.

        Parameters:
            x (torch.Tensor): input values
            hidden (torch.Tensor): current hidden state
            ts (int): current hidden timestep

        Returns:
            hidden (torch.Tensor): a new hidden state
        """
        x = torch.cat([x, hidden], 1)
        x = self.backbone(x)

        g_out = self.g_head(x)
        h_out = self.h_head(x)

        return self._new_hidden(x, g_out, h_out, ts)
