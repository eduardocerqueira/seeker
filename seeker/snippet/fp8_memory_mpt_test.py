#date: 2023-12-08T16:45:06Z
#url: https://api.github.com/gists/a7f1d7f04d105615277183fb0bfb0cc0
#owner: https://api.github.com/users/cli99

import datetime
import gc
import pathlib
import torch

from composer.utils import get_device
from omegaconf import OmegaConf as om
from llmfoundry.models.mpt.modeling_mpt import ComposerMPTCausalLM
from composer.core import Precision
from composer import Trainer

import transformer_engine.pytorch as te
from transformer_engine.common import recipe

BATCH_SIZE = 1
D_MODEL = 8192
SEQ_LEN = 4096
NUM_LAYERS = 4

USE_TE = True
SHARDING_STRATEGY = 'FULL_SHARD'


def main():
    trace_dir = pathlib.Path(__file__).parent.joinpath("traces")
    trace_dir.mkdir(exist_ok=True)
    now = datetime.datetime.now().strftime("%Y_%m_%d:%H.%M.%S")

    precision_config = {
        'fp8_format': recipe.Format.HYBRID,
        'amax_history_len': 16,
        'amax_compute_algo': 'max',
    }
    fp8_recipe = recipe.DelayedScaling(**precision_config)

    device = get_device('gpu')
    fc_type = 'te' if USE_TE else 'torch'
    model_cfg = {
        'name': 'mpt_causal_lm',
        'd_model': D_MODEL,
        'n_heads': 64,
        'n_layers': NUM_LAYERS,
        'expansion_ratio': 4,
        'max_seq_len': SEQ_LEN,
        'vocab_size': 100352,
        'attn_config': {
            'attn_impl': 'triton',
            'attn_type': 'grouped_query_attention',
            'kv_n_heads': 8,
        },
        'fc_type': fc_type,
        'no_bias': True,
    }

    model_cfg = om.create(model_cfg)

    fsdp_config = {
        'sharding_strategy': SHARDING_STRATEGY,
        'mixed_precision': 'PURE',
        'forward_prefetch': False,
        'backward_prefetch': 'BACKWARD_POST',
        'limit_all_gathers': True,
        'forward_prefetch_limit': 2,
        'backward_prefetch_limit': 2,
    }

    model = ComposerMPTCausalLM(model_cfg)
    model = device.module_to_device(model)

    trainer = Trainer(
        model=model,
        device='gpu',
        fsdp_config=fsdp_config,
        precision=Precision.AMP_BF16 if not USE_TE else Precision.AMP_FP8,
    )
    gc.collect()

    def trace_handler(prof):
        torch.cuda.empty_cache()
        gc.collect()
        prof.export_memory_timeline(str(trace_dir.joinpath(f"{now}_{USE_TE}_{SHARDING_STRATEGY}.html")),
                                    torch.cuda.current_device())

    x = {
        'input_ids': torch.ones((
            BATCH_SIZE,
            SEQ_LEN,
        ), device="cuda", dtype=torch.int64),
        'labels': torch.ones((BATCH_SIZE, SEQ_LEN), device="cuda", dtype=torch.int64)
    }

    with torch.profiler.profile(schedule=torch.profiler.schedule(skip_first=0, wait=1, warmup=1, active=3, repeat=1),
                                on_trace_ready=trace_handler,
                                record_shapes=True,
                                profile_memory=True,
                                with_stack=True) as p:

        for i in range(6):
            if USE_TE:
                with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    outputs = trainer.state.model(x)
                    loss = trainer.state.model.loss(outputs, x)
                    loss.backward()
            else:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = trainer.state.model(x)
                    loss = trainer.state.model.loss(outputs, x)
                    loss.backward()

            max_memory_allocated = torch.cuda.max_memory_allocated()
            max_memory_reserved = torch.cuda.max_memory_reserved()
            print(
                f'iter: {i}, max_memory_allocated: {max_memory_allocated}, max_memory_reserved: {max_memory_reserved}')
            p.step()
            torch.cuda.reset_peak_memory_stats()


# python -m composer.cli.launcher -n 8 --master_port 26000 fp8_memory_mpt_test.py

if __name__ == "__main__":
    main()
