#date: 2025-10-31T17:09:27Z
#url: https://api.github.com/gists/d52e22338448652e9b1ad0fc309dc158
#owner: https://api.github.com/users/yoavartzi

# %%
import torch
import timeit
import matplotlib.pyplot as plt


def bench(dev, batch_size, hidden_size):
  x = torch.ones(batch_size, 5, device=dev)
  model = torch.nn.Sequential(
    torch.nn.Linear(5, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, 1),
  )
  model.to(dev)
  return timeit.timeit(lambda: model(x), number=1000)


# %%

hidden_size = 1024
bs_l = [2 ** i for i in range(10)]
out = {}
for d in ("mps", "cpu"):
  dev = torch.device(d)
  bs_time_l = [bench(dev, bs, hidden_size) for bs in bs_l]
  out[d] = bs_time_l

fig = plt.figure()
for d in out:
  plt.plot(bs_l, out[d], label=d)
plt.legend()
plt.xlabel("Batch size")
plt.ylabel("Time (s)")
plt.xscale("log")
plt.title(f"Batch size vs. time for hidden size {hidden_size}")