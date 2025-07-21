#date: 2025-07-21T17:08:04Z
#url: https://api.github.com/gists/f3bfa1cdfe95e5ad8df005a21a337e4b
#owner: https://api.github.com/users/dhbrojas

import torch
from tqdm import tqdm
from torch.nn import Module
from torch.nn.functional import cross_entropy
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
)

BATCH = 16
SEQUENCE_LENGTH = 2048
ITERS = 16
WARMUP_ITERS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = AutoConfig.from_pretrained("./qwen3-0.6b")
model = AutoModelForCausalLM.from_config(config)
model = model.to(device=device, dtype=torch.bfloat16)
model.train()


@torch.compile
def causal_lm_forward_backward(model: Module) -> int:
    x = torch.randint(0, 1000, (BATCH, SEQUENCE_LENGTH), device=device)
    y = torch.randint(0, 1000, (BATCH, SEQUENCE_LENGTH), device=device)

    logits = model(input_ids=x).logits
    loss = cross_entropy(logits.view(BATCH*SEQUENCE_LENGTH, -1), y.view(-1), reduction='none')
    loss.mean().backward()

    return x.numel()


tokens = "**********"
timings = []

stream = torch.cuda.Stream()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


for i in tqdm(range(ITERS), desc="Training"):
    with torch.cuda.stream(stream):  # type: ignore
        start.record(stream)
        num_tokens = "**********"
        end.record(stream)

    torch.cuda.synchronize()
    tokens.append(num_tokens)
    timings.append(start.elapsed_time(end))

# Remove the first few measurements
tokens, timings = tokens[WARMUP_ITERS: "**********":]
total_num_tokens = "**********"

print(f"Highest GPU Memory Usage: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f}GB")
print(f"Mean forward/backward time: {round(sum(timings) / len(timings))}ms")
print(f"Therotical TPS (per GPU): "**********":,} tokens/s")((total_num_tokens / sum(timings))*1000):,} tokens/s")