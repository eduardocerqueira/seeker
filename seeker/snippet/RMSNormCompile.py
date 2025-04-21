#date: 2025-04-21T17:00:09Z
#url: https://api.github.com/gists/439940a38e62b377b4bcf5a8e1b719bb
#owner: https://api.github.com/users/ZejiaZheng

import torch
import time

def naive_rmsnorm(x, eps=1e-4):
    """Reference implementation of RMSNorm."""
    dim = -1
    mean_sq = torch.mean(x**2, dim=dim, keepdim=False, dtype=torch.float32)
    scale = torch.rsqrt(mean_sq + eps)
    scale = scale.unsqueeze(dim)
    output = x.to(torch.float32) * scale
    return output.to(x.dtype), scale

dtype = torch.float16
n = 512
shape = (2*1024*1024*4, n)
x = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=False)
eps = 1e-4
atol = 1e-3
rtol = 1e-3

for i in range(2):
    out_naive, scale = naive_rmsnorm(x, eps)
torch.cuda.synchronize()

# Naive implementation. 
start_time = time.time()
for i in range(100):
    out_naive, scale = naive_rmsnorm(x, eps)
torch.cuda.synchronize()
end_time = time.time()
print(f"Naive RMSNorm time 100 iterations: {end_time - start_time:.4f} seconds")

# Compile.
naive_rmsnorm_strict = torch.compile(naive_rmsnorm)
out_naive, scale = naive_rmsnorm_strict(x, eps)
torch.cuda.synchronize()
start_time = time.time()
for i in range(100):
    out_naive, scale = naive_rmsnorm_strict(x, eps)
torch.cuda.synchronize()
end_time = time.time()
print(f"Non dynamimc compiled RMSNorm time 100 iterations: {end_time - start_time:.4f} seconds")

# Dynamic = True. 
naive_rmsnorm_dynamic = torch.compile(naive_rmsnorm, dynamic=True)
out_naive, scale = naive_rmsnorm_dynamic(x, eps)
torch.cuda.synchronize()
start_time = time.time()
for i in range(100):
    out_naive, scale = naive_rmsnorm_dynamic(x, eps)
torch.cuda.synchronize()
end_time = time.time()
print(f"Dynamimc compiled RMSNorm time 100 iterations: {end_time - start_time:.4f} seconds")