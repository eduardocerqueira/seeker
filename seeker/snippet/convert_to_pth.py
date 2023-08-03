#date: 2023-08-03T16:54:35Z
#url: https://api.github.com/gists/0e0317755dc126ec31614222b05bac5d
#owner: https://api.github.com/users/wfjsw

import pathlib
import torch
from safetensors import safe_open
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("input", metavar="in", type=str, default="model.safetensors", help="Input .safetensors model path")
parser.add_argument("-m", "--metadata", required=False, metavar="meta", type=str, default="model.json", help="Metadata .json path")
parser.add_argument("output", metavar="out", nargs="?", default=None, type=str, help="Output .pth model path")
cmd_opts = parser.parse_args()

safetensors_path = cmd_opts.input
metadata_path = cmd_opts.metadata if cmd_opts.metadata is not None else safetensors_path.replace('.safetensors', '.json')
pth_path = cmd_opts.output if cmd_opts.output is not None else safetensors_path.replace('.safetensors', '.pth')

state_dict = {}
metadata = {}

with safe_open(safetensors_path, 'rb') as f:
    for k in f.keys():
        state_dict[k] = f.get_tensor(k)

    if not pathlib.Path(metadata_path).exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        meta = f.metadata()
        metadata['config'] = json.loads(meta.get('config'))
        metadata['sr'] = meta.get('sr')
        metadata['f0'] = 1 if meta.get('f0') == 'true' else 0
        metadata['version'] = meta.get('version', 'v1')
        metadata['info'] = meta.get('info')

metadata['weight'] = state_dict

torch.save(metadata, pth_path)
