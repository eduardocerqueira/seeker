#date: 2023-05-26T16:45:05Z
#url: https://api.github.com/gists/a947131f6c3615860632fcbb584008d0
#owner: https://api.github.com/users/iszotic

# Script for converting a HF Diffusers saved pipeline to a Stable Diffusion checkpoint.
# *Only* converts the ControlNet part.
# Does not convert optimizer state or any other thing.

import argparse
import os.path as osp
import re
import csv


import torch
from safetensors.torch import load_file, save_file


# =================#
# UNet Conversion #
# =================#
unet_conversion_map = []
#brute force conversion
with open('controlnet.csv', newline='') as csvfile:
    conv = csv.reader(csvfile)
    for row in conv:
        unet_conversion_map.append((row[0], row[1]))

def convert_unet_state_dict(unet_state_dict):
    # buyer beware: this is a *brittle* function,
    # and correct output requires that all of these pieces interact in
    # the exact order in which I have arranged them.
    mapping = {k: k for k in unet_state_dict.keys()}
    for sd_name, hf_name in unet_conversion_map:
        mapping[hf_name] = sd_name
    new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items()}
    return new_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    required = False
    parser.add_argument("--model_path", default=None, type=str, required=required, help="Path to the model to convert.")
    parser.add_argument("--checkpoint_path", default=None, type=str, required=required, help="Path to the output model.")
    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    parser.add_argument(
        "--use_safetensors", action="store_true", help="Save weights use safetensors, default is ckpt."
    )
    args = parser.parse_args()
    
    assert args.model_path is not None, "Must provide a model path!"

    assert args.checkpoint_path is not None, "Must provide a checkpoint path!"

    # Path for safetensors
    unet_path = osp.join(args.model_path, 'controlnet', "diffusion_pytorch_model.safetensors")

    # Load models from safetensors if it exists, if it doesn't pytorch
    if osp.exists(unet_path):
        unet_state_dict = load_file(unet_path, device="cpu")
    else:
        unet_path = osp.join(args.model_path, 'controlnet', "diffusion_pytorch_model.bin")
        unet_state_dict = torch.load(unet_path, map_location="cpu")

    # Convert the UNet model with ControlNet
    unet_state_dict = convert_unet_state_dict(unet_state_dict)
    unet_state_dict = {"control_model." + k: v for k, v in unet_state_dict.items()}

    # Put together new checkpoint
    state_dict = unet_state_dict
    if args.half:
        state_dict = {k: v.half() for k, v in state_dict.items()}

    if args.use_safetensors:
        save_file(state_dict, args.checkpoint_path)
    else:
        state_dict = {"state_dict": state_dict}
        torch.save(state_dict, args.checkpoint_path)