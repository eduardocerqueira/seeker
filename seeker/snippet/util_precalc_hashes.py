#date: 2023-01-16T16:35:34Z
#url: https://api.github.com/gists/4f56fdd1dd0ab207c6231eef3006c012
#owner: https://api.github.com/users/catboxanon

import argparse
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO 
from tqdm import tqdm
import modules.hashes

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True)
parser.add_argument('-t', '--type', type=str, required=True, choices=['checkpoint', 'hypernet'])
args = parser.parse_args()

class NullIO(StringIO):
	def write(self, txt):
		pass

def silent(fn):
	def silent_fn(*args, **kwargs):
		with redirect_stdout(NullIO()):
			return fn(*args, **kwargs)
	return silent_fn

silent_sha256 = silent(modules.hashes.sha256)

if __name__ == '__main__':
	for file in tqdm([f for f in Path(args.path).glob('**/*') if f.is_file() and f.suffix in ['.ckpt', '.safetensors', '.pt']]):
		if (
			(args.type == 'checkpoint' and file.suffix in ['.ckpt', '.safetensors'])
			or (args.type == 'hypernet' and file.suffix in ['.pt'])
		):
			title = f'{args.type}/{file.stem if args.type in ["hypernet"] else file.name}'
			silent_sha256(str(file), title)