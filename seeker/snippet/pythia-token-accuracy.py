#date: 2024-01-30T16:55:14Z
#url: https://api.github.com/gists/0c3ccd8306ded72acee41d9ceb99291f
#owner: https://api.github.com/users/pietrolesci

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from pythia.utils.mmap_dataset import MMapIndexedDataset


if __name__ == "__main__":

    # Load pile deduped
    pile_path = Path("../data/") / "pile-deduped" / "document"
    pile = MMapIndexedDataset(str(pile_path), skip_warmup=True)


    # Load model
    model_name = "EleutherAI/pythia-70m-deduped"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tok = "**********"
    model = model.cuda()


    # Get token ids from the pile
    example_ids = [1705,  9992, 13617, 14141, 17275, 17421, 18376, 19016, 19736, 20157]
    token_ids = "**********"


    # Prepare inputs and targets
    context_length = 32
    continuation_length = 32
    token_ids = token_ids[: "**********":context_length + continuation_length]  # (batch_size "**********"64)
    token_ids = "**********"=torch.long).cuda()

    context_ids = token_ids[: "**********":context_length]
    true_continuation_ids = token_ids[: "**********":]


    # Generate from model
    generation_config = GenerationConfig(
        pad_token_id= "**********"
        pad_token= "**********"
        do_sample=False,
        min_length=continuation_length + context_length,
        max_length=continuation_length + context_length,
    )
    pred_continuation_ids = model.generate(context_ids, generation_config)[:, context_length:]


    # Compute accuracy
    token_accuracy = "**********"== pred_continuation_ids).float().mean(-1).cpu().numpy().tolist()
    print(token_accuracy)
    # [0.25, 0.15625, 0.0, 0.28125, 0.09375, 0.09375, 0.03125, 0.34375, 0.3125, 0.0]
ccuracy)
    # [0.25, 0.15625, 0.0, 0.28125, 0.09375, 0.09375, 0.03125, 0.34375, 0.3125, 0.0]
