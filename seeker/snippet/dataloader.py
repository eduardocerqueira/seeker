#date: 2025-07-22T17:09:50Z
#url: https://api.github.com/gists/44607447cc7a7d66451561f2b1279347
#owner: https://api.github.com/users/navmarri14


from roloader import StreamingDataLoader, StreamingDataset

import pickle
import torch
# function to load a list of rows give a local object path
def get_rows(object_path):
    loaded_data = pickle.load(open(object_path, "rb"))
    loaded_data['prompt'] = loaded_data['prompt'].prompt
    return [loaded_data]


def collate_fn(data):
    prompts = [d['prompt'] for d in data]
    uuids = [d['uuid'] for d in data]
    outputs = torch.stack([d['output_ids'] for d in data])
    all_logits = torch.stack([d['all_logits'] for d in data])

    return {
        'prompts': prompts,
        'uuids': uuids,
        'outputs': outputs,
        'all_logits': all_logits,
    }

# S3 location of the index created in 2.
input_dir = "s3://3dfm-data/users/marri/shape-vae/speculative-decoding/ray-infer/data/42M/H200/infer-single-call/"

import sys
import types
import pickle
from dataclasses import dataclass

@dataclass
class PromptWithMetadata:
    prompt: str
    bbox: list[float] | None = None

# 2. Create fake module hierarchy: shape_vae.eval.utils
shape_vae = types.ModuleType("shape_vae")
eval_mod = types.ModuleType("shape_vae.eval")
utils_mod = types.ModuleType("shape_vae.eval.utils")

# 3. Attach the class to utils_mod
setattr(utils_mod, "PromptWithMetadata", PromptWithMetadata)

# 4. Wire up the fake modules
sys.modules["shape_vae"] = shape_vae
sys.modules["shape_vae.eval"] = eval_mod
sys.modules["shape_vae.eval.utils"] = utils_mod

dataset = StreamingDataset(
    input_dir=input_dir,
    cache_dir="/dev/shm/.cache/",
    cache_max_size_bytes=200_000_000_000,
    memory_max_size_bytes=160_000_000_000,
    num_download_threads=3,
    num_load_threads=3,
    row_getter_func=get_rows,
)


dataloader = StreamingDataLoader(
    dataset,
    shuffle=True,
    seed=8,
    batch_size=16,
    collate_fn=collate_fn,
    # num_workers=8,
)
import time

if __name__ == "__main__":
    end = None
    time_start = time.time()
    for idx, batch in enumerate(dataloader):
        # start = time.time()
        # if end is not None:
        #     print(f"Time taken for loading batch {idx}: {end - start} seconds")
        # print(f"Size of logits {batch['all_logits'].shape}, \
        #         size of output_ids {batch['outputs'].shape}, \
        #         size of prompts {len(batch['prompts'])}, \
        #         size of uuids {len(batch['uuids'])}")
        end = time.time()
        print(f"Time taken for processing batch {idx}: {round(end - time_start, 2)} seconds")
        
