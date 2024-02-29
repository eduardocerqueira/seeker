#date: 2024-02-29T16:51:55Z
#url: https://api.github.com/gists/fbe9027de0c4aeff151150c4d6479229
#owner: https://api.github.com/users/kavishka-dot

step = 0.05
for i in range(20):
  #forward pass
  ypred = [n(x) for x in xs]
  loss = sum((yout-ygt)**2 for ygt , yout in zip(ypred,ys))

  #backward pass
  loss.backward()

  #parameter updates
  for p in n.parameters():
    p.data += (-1)*(step) * p.grad