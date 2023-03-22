#date: 2023-03-22T17:09:46Z
#url: https://api.github.com/gists/8c218f91629fe8eecedaddeb04bd1c36
#owner: https://api.github.com/users/corrosivelogic

import torch.optim as optim
epsilon = 2./255

delta = torch.zeros_like(pig_tensor, requires_grad=True)
opt = optim.SGD([delta], lr=1e-1)

for t in range(30):
    pred = model(norm(pig_tensor + delta))
    loss = -nn.CrossEntropyLoss()(pred, torch.LongTensor([341]))
    if t % 5 == 0:
        print(t, loss.item())
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    delta.data.clamp_(-epsilon, epsilon)
    max_class = pred.max(dim=1)[1].item()