#date: 2023-03-01T16:56:55Z
#url: https://api.github.com/gists/5d5ce8dd764b89982bbfbcc03ac9f72e
#owner: https://api.github.com/users/takuma104

import torch
import torch.nn.functional as F
import safetensors.torch
import sys


def load_checkpoint(fn):
    if fn.endswith(".safetensors"):
        checkpoint = safetensors.torch.load_file(fn)
    else:
        checkpoint = torch.load(fn)
    return checkpoint


if __name__ == "__main__":
    fn_a = sys.argv[1]
    fn_b = sys.argv[2]

    checkpoint_a = load_checkpoint(fn_a)
    checkpoint_b = load_checkpoint(fn_b)

    all_keys = set(checkpoint_a.keys()) | set(checkpoint_b.keys())
    all_keys = sorted(list(all_keys))

    print("diff,key,shape,type,cossim")
    for k in all_keys:
        a = checkpoint_a[k] if k in checkpoint_a else None
        b = checkpoint_b[k] if k in checkpoint_b else None
        if a is not None and b is not None:
            if a.shape != b.shape:
                print(f'*,{k},"{tuple(a.shape)} & {tuple(b.shape)}",-,-')
            elif a.dtype != b.dtype:
                print(f'*,{k},"{tuple(a.shape)}","{str(a.dtype)} & {str(b.dtype)}",-')
            else:
                if torch.equal(a, b):
                    print(f'=,{k},"{tuple(a.shape)}",{str(a.dtype)},-')
                else:
                    cossim = F.cosine_similarity(
                        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
                    )[0]
                    print(f'*,{k},"{tuple(a.shape)}",{str(a.dtype)},{cossim:.4f}')
        elif b is None:
            print(f'-,{k},"{tuple(a.shape)}",{str(a.dtype)},-')
        else:
            print(f'+,{k},"{tuple(b.shape)}",{str(b.dtype)},-')
