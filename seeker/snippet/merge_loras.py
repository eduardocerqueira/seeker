#date: 2025-06-12T17:13:51Z
#url: https://api.github.com/gists/fc7b80f58562ced0bc18d6a57b6cd0c1
#owner: https://api.github.com/users/jaretburkett

import gc
from collections import OrderedDict
import os

# leave in this if for autoformatting purposes
if True:
    import torch
    from safetensors.torch import load_file, save_file


def flush():
    torch.cuda.empty_cache()
    gc.collect()


metadata = OrderedDict()
metadata["format"] = "pt"

# you can add as many as you want. Be sure to adjust the weights accordingly 1.0 is full weight. 
lora_paths = [
    ("/path/to/lora.safetensors", 0.2),
    ("/path/to/lora.safetensors", 0.2),
    ("/path/to/lora.safetensors", 0.2),
]
output_path = "/path/to/save.safetensors"
dtype = torch.bfloat16

device = torch.device("cpu")

output_state_dict = {}


def pad_tensor(tensor, target_shape):
    current_shape = tensor.shape
    padding = []
    for i in range(len(current_shape) - 1, -1, -1):
        if i < len(target_shape):
            padding.extend([0, max(0, target_shape[i] - current_shape[i])])
        else:
            padding.extend([0, 0])
    return torch.nn.functional.pad(tensor, padding)


for idx, (lora_path, multiplier) in enumerate(lora_paths):
    print(f"Loading LoRA {idx + 1}/{len(lora_paths)}")
    lora_state_dict = load_file(lora_path)

    for key, value in lora_state_dict.items():
        value = value.to(torch.float32) * multiplier
        if key not in output_state_dict:
            output_state_dict[key] = value
        else:
            target_shape = torch.max(torch.tensor(
                output_state_dict[key].shape), torch.tensor(value.shape))
            output_state_dict[key] = pad_tensor(
                output_state_dict[key], target_shape)
            value = pad_tensor(value, target_shape)
            output_state_dict[key] += value

    flush()

for key, value in output_state_dict.items():
    output_state_dict[key] = value.to('cpu', dtype)

print("Saving model...")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
save_file(output_state_dict, output_path, metadata=metadata)

print(f"Successfully saved merge to to {output_path}")
print("Done.")
