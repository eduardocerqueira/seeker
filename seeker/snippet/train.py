#date: 2026-02-17T17:36:33Z
#url: https://api.github.com/gists/8c44efc7478b084c9a56761a2b737a7c
#owner: https://api.github.com/users/Xayaan

"""
High MFU training implementation for Crusades by Dr Big Brain
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


_COMPILED = False
_compiled_train_step = None


@dataclass
class InnerStepsResult:

    final_logits: torch.Tensor  # Output logits from last forward pass
    total_tokens: "**********"
    final_loss: float  # Loss value from last training step


def _train_step_fn(model, input_ids, labels):

    outputs = model(input_ids)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )
    return logits, loss


def inner_steps(model, data_iterator, optimizer, num_steps, device):

    global _COMPILED, _compiled_train_step

    if not _COMPILED:
        _compiled_train_step = torch.compile(
            _train_step_fn,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

        if torch.cuda.is_available():
            model.forward = torch.compile(
                model.forward,
                mode="max-autotune",
                fullgraph=False,
                dynamic=False,
            )

        _COMPILED = True

    total_tokens = "**********"
    final_logits = None
    final_loss = 0.0

    for step in range(num_steps):
        batch = next(data_iterator)
        batch = batch.to(device, dtype=torch.long)

        input_ids = batch[:, :-1]
        labels = batch[:, 1:]

        logits, loss = _compiled_train_step(model, input_ids, labels)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += "**********"
        final_logits = logits.detach().float()
        final_loss = loss.item()

    return InnerStepsResult(
        final_logits=final_logits,
        total_tokens= "**********"
        final_loss=final_loss,
    )
_loss=final_loss,
    )
