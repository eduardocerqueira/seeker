#date: 2023-03-29T17:22:38Z
#url: https://api.github.com/gists/047b2bf4cb01f60d4b438a08f3acb6e4
#owner: https://api.github.com/users/anijain2305

import torch
import torch.overrides

class PrintFunctionMode(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs={}):
        print(func)
        if func is torch.cuda.get_rng_state:
            return torch.zeros(4)
        if func is torch.cuda.set_rng_state:
            return
        return func(*args, **kwargs)
        # return func(*args, **kwargs)

class Custom(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        state = torch.cuda.get_rng_state()
        ctx.save_for_backward(x, state)
        torch.cuda.set_rng_state(state)
        return torch.sin(x)
    
    @staticmethod
    def backward(ctx, grad_out):
        x, state = ctx.saved_tensors
        torch.cuda.set_rng_state(state)
        return grad_out * torch.sigmoid(x)


custom = Custom.apply
    
x = torch.rand(4, device="cuda", requires_grad=True)
def dispatcher_get(device="cuda"):
    # print("GET RNG")
    return (device,)

def dispatcher_set(x):
    # print("SET RNG")
    return (x,)

old_set_rng_state = torch.cuda.set_rng_state
old_get_rng_state = torch.cuda.get_rng_state
torch.cuda.set_rng_state = torch.overrides.wrap_torch_function(dispatcher_set)(torch.cuda.set_rng_state)
torch.cuda.get_rng_state = torch.overrides.wrap_torch_function(dispatcher_get)(torch.cuda.get_rng_state)

with PrintFunctionMode():
    y = custom(x).sum().backward()
        
torch.cuda.set_rng_state = old_set_rng_state
torch.cuda.get_rng_state = old_get_rng_state
    
    
