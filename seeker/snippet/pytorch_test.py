#date: 2025-04-14T16:36:53Z
#url: https://api.github.com/gists/310d7dc82acbb84c682ca28c1740cd52
#owner: https://api.github.com/users/ericspod

import torch

print("Version=",torch.__version__)
print("CUDA=",torch.cuda.is_available())

for i in range(torch.cuda.device_count()):
  props=torch.cuda.get_device_properties(i)
  print(f"  {props.name}, mem={int(props.total_memory/2**20)}MiB")
  