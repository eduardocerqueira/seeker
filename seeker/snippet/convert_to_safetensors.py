#date: 2023-08-03T16:54:35Z
#url: https://api.github.com/gists/0e0317755dc126ec31614222b05bac5d
#owner: https://api.github.com/users/wfjsw

import torch
from safetensors.torch import save_file
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("input", metavar="in", type=str, default="model.pth", help="Input .pth model path")
parser.add_argument("output", metavar="out", nargs="?", default=None, type=str, help="Output .safetensors model path")
parser.add_argument("-m", "--metadata", metavar="meta", required=False, default=None, type=str, help="Output .json metadata path")
cmd_opts = parser.parse_args()

pth_path = cmd_opts.input
safetensors_path = cmd_opts.output if cmd_opts.output is not None else pth_path.replace('.pth', '.safetensors')
metadata_path = cmd_opts.metadata if cmd_opts.metadata is not None else safetensors_path.replace('.safetensors', '.json')

model = torch.load(pth_path)
state_dict = model.get('weight')
config = model.get('config')

metadata_safetensors = {
    'config': json.dumps(config),
    'sr': model.get('sr'),
    'f0': 'true' if model.get('f0') else 'false',
    'version': model.get('version', 'v1'),
    'info': model.get('info')
}

metadata = {k:v for k,v in model.items() if k != 'weight'}

save_file(state_dict, safetensors_path, metadata_safetensors)
with open(metadata_path, 'w') as f:
    json.dump(metadata, f)
