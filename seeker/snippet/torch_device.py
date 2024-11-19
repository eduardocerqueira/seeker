#date: 2024-11-19T16:50:48Z
#url: https://api.github.com/gists/5b558e0a552627efb1c2d54b90a7cec6
#owner: https://api.github.com/users/vadim0x60

def torch_device():
    import torch
    t = torch.Tensor([0])
    for device in ['xla', 'cuda', 'mps', 'xpu', 'cpu']:
        try:
            t.to(device)
            return torch.device(device)
        except RuntimeError:
            pass