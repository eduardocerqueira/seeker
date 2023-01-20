#date: 2023-01-20T17:07:47Z
#url: https://api.github.com/gists/18d6f29b0966462c0b8a54ef814af413
#owner: https://api.github.com/users/jaykru

#!/usr/bin/env python3
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

words = open('names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# set up example trigrams
xs, ys, zs = [],[],[]
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(chs,chs[1:],chs[2:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        ix3 = stoi[ch3]
        xs.append(ix1)
        ys.append(ix2)
        zs.append(ix3)
xs,ys,zs = torch.tensor(xs), torch.tensor(ys), torch.tensor(zs)
num = xs.nelement()
print("num of examples: ", num)

# randomly initialize 27 neurons' weights. each neuron receives 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27, 27), generator=g, requires_grad=True)

# gradient descent
for k in range(20):
  # forward pass
  # xenc = F.one_hot(xs, num_classes=27).float()
  # input to the network: one-hot encoding
  logits = W[xs]  # predict log-counts; (N,27) * (27,27,27) = (N,27,27)
  counts = logits.exp() # counts, equivalent to N
  probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
  loss = -probs[torch.arange(num), ys, zs].log().mean() + 0.01*(W**2).mean()
  print(loss.item())

  # backward pass
  W.grad = None # set to zero the gradient
  loss.backward()

  # update
  W.data += -50 * W.grad

# finally, sample from the 'neural net' model
g = torch.Generator()
g.manual_seed(2147483655647)

for i in range(10):
  out = []
  ix = 0
  iy = 0
  while True:
    xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
    logits = W[ix] # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    p = counts / counts.sum(1, keepdims=True) # probabilities for next two characters
    sel = torch.multinomial(p.view(27**2), num_samples=1, replacement=True, generator=g)
    (ix,iy) = (sel//27, sel%27)

    out.append(itos[ix.item()])
    if ix == 0:
      break
    out.append(itos[iy.item()])
    if iy == 0:
      break
    ix = iy.item()
  print(''.join(out))
