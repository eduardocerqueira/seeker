#date: 2025-05-21T16:57:09Z
#url: https://api.github.com/gists/271d55239a4ab679d9754f2796f0f8d7
#owner: https://api.github.com/users/Aceticia

"""
Performance Benchmark for Flex Attention vs SDPA

This script benchmarks the performance of PyTorch's Flex Attention implementation against
the standard Scaled Dot Product Attention (SDPA) for various attention patterns.

Key Features:
- Tests multiple attention patterns: block causal, sliding window, same sensor, and combinations
- Compares both compiled and non-compiled versions
- Measures execution time across multiple runs
- Supports multi-sensor data format (B, S, T, D) where:
  - B: Batch size
  - S: Number of sensors
  - T: Time steps
  - D: Feature dimension

Usage:
    Simply run the script to see performance comparisons between flex_attention and SDPA
    for different attention patterns and compilation settings.

Dependencies:
    - torch
    - einx
    - pandas
"""

import time

import einx
import pandas as pd
import torch
from torch.nn.attention.flex_attention import (
    and_masks,
    create_block_mask,
    flex_attention,
)

repeats = 100
outputs = []

for B, S, H, T, D in [(4, 64, 8, 128, 64)]:
    window_size = 10

    # Make a fake query, key, value
    query = torch.randn(B, H, S * T, D).cuda()
    key = torch.randn(B, H, S * T, D).cuda()
    value = torch.randn(B, H, S * T, D).cuda()

    # indicators
    sensor_indicator = einx.rearrange("s -> b (s t)", torch.arange(S), b=B, t=T).cuda()
    time_start_indicator = einx.rearrange(
        "t -> b (s t)", torch.arange(T), b=B, s=S
    ).cuda()
    time_end_indicator = time_start_indicator + 1

    def make_block_causal_mask(b, h, q_idx, kv_idx):
        # Tokens can attend to other tokens that end earlier than it starts
        return time_end_indicator[b, kv_idx] <= time_start_indicator[b, q_idx] + 1

    def make_sliding_attention_mask(b, h, q_idx, kv_idx):
        return (
            time_start_indicator[b, kv_idx] - time_start_indicator[b, q_idx]
        ).abs() < window_size

    def make_same_sensor_mask(b, h, q_idx, kv_idx):
        return sensor_indicator[b, kv_idx] == sensor_indicator[b, q_idx]

    def make_mask(mask_type):
        maker_fn = {
            "block_causal": make_block_causal_mask,
            "sliding": make_sliding_attention_mask,
            "same_sensor": make_same_sensor_mask,
            "block_causal_and_sliding": and_masks(
                make_block_causal_mask, make_sliding_attention_mask
            ),
            "block_causal_and_same_sensor": and_masks(
                make_block_causal_mask, make_same_sensor_mask
            ),
            "all": and_masks(
                make_block_causal_mask,
                make_sliding_attention_mask,
                make_same_sensor_mask,
            ),
        }[mask_type]
        return create_block_mask(maker_fn, B=B, H=None, Q_LEN=S * T, KV_LEN=S * T)

    def run_attention(mask, attention_type, query, key, value, sdpa, flex):
        if attention_type == "sdpa":
            return sdpa(query, key, value, attn_mask=mask)
        elif attention_type == "flex":
            return flex(query, key, value, block_mask=mask)

    # Compile always helps
    for compile in [True]:
        for mask_type in [
            "full",
            "block_causal",
            "sliding",
            "same_sensor",
            "block_causal_and_sliding",
            "block_causal_and_same_sensor",
            "all",
        ]:

            if compile:
                sdpa = torch.compile(torch.nn.functional.scaled_dot_product_attention)
                flex = torch.compile(flex_attention)
            else:
                sdpa = torch.nn.functional.scaled_dot_product_attention
                flex = flex_attention

            if mask_type == "full":
                mask = None
            else:
                mask = make_mask(mask_type)
            for attention_type in ["sdpa", "flex"]:
                # Random attention mask for SDPA. They always take the same amount of time
                if mask is not None and attention_type == "sdpa":
                    mask_this = torch.randn(B, 1, S * T, S * T, device=query.device)
                else:
                    mask_this = mask

                # Warm up
                run_attention(mask_this, attention_type, query, key, value, sdpa, flex)

                # Run
                torch.cuda.reset_peak_memory_stats()
                start = time.time()
                for _ in range(repeats):
                    run_attention(
                        mask_this, attention_type, query, key, value, sdpa, flex
                    )
                end = time.time()
                peak_memory_mb = (
                    torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024**2
                )
                outputs.append(
                    {
                        "mask_type": mask_type,
                        "attention_type": attention_type,
                        "time": (end - start) / repeats,
                        "compile": compile,
                        "B": B,
                        "S": S,
                        "H": H,
                        "T": T,
                        "D": D,
                        "peak_memory_mb": peak_memory_mb,
                    }
                )
                print(
                    f"{compile=}, {mask_type=}, {attention_type=}, time: {(end - start) / repeats}, peak_memory_mb: {peak_memory_mb}"
                )

outputs = pd.DataFrame(outputs)
outputs.to_csv("outputs.csv", index=False)
print(outputs)
