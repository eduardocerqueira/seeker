#date: 2024-03-11T16:56:50Z
#url: https://api.github.com/gists/f414f2db2b010796a8613988c81cf300
#owner: https://api.github.com/users/kavishka-dot

x = train_data[:block_size] # train examples
y = train_data[1:block_size+1] # targets

for t in range(block_size):
  context = x[:t+1]
  target = y[t]
  print(f"when input is {context} the target: {target}")