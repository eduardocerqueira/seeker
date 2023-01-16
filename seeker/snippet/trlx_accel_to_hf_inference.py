#date: 2023-01-16T16:49:53Z
#url: https://api.github.com/gists/a6e1da299a075fc2db3b8ff637ab0dd0
#owner: https://api.github.com/users/jon-tow

"""This scripts shows how to convert a `accelerate` checkpoint to a `hf` model
that can be used for inference.

NOTE: You may need to call this script with `accelerate launch` (or the proper distributed launcher)
to run it on multiple GPUs and load the model properly (e.g. when the model was trained with `deepspeed`). 
"""
import argparse
import yaml
import trlx
from trlx.data.configs import TRLConfig


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/ppo_config.yml")
parser.add_argument("--checkpoint", type=str, default="ckpts")
args = parser.parse_args()


config = TRLConfig.load_yaml(yaml.safe_load(args.config))


# Convert `accelerate` checkpoint states to hf model

trainer = trlx.trainer.accelerate_ppo_trainer.AcceleratePPOTrainer(config=config)

trlx.utils.print_rank_0("[Info] Loading trainer state from checkpoint")
trainer.load(args.checkpoint)

trlx.utils.print_rank_0("[Info] Saving trainer state with `save_pretrained` for easy loading with `from_pretrained`")
trainer.save_pretrained(args.checkpoint)  # NOTE: This is currently unsupported for ILQL

trlx.utils.print_rank_0("[Info] Loading model `from_pretrained`")
# NOTE: `save_pretrained` models are saved into a `hf_model` sub-directory of the checkpoint dir
model = trainer.model.base_model.from_pretrained(f"{args.checkpoint}/hf_model")


# Generate text with the loaded model

trlx.utils.print_rank_0("[Info] Generating text with loaded model...")
inputs = "**********"
    ["The brown fox jumped over "],
    return_tensors="pt",
)
raw_outputs = model.generate(**inputs, **config.method.gen_kwargs)
text = "**********"=True)
trlx.utils.print_rank_0(text)
 skip_special_tokens=True)
trlx.utils.print_rank_0(text)
