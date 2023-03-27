#date: 2023-03-27T16:55:43Z
#url: https://api.github.com/gists/ca52d0f6ad31454370c2af4fdfb70803
#owner: https://api.github.com/users/ShawonAshraf

import pynvml

pynvml.nvmlInit()

device_count = pynvml.nvmlDeviceGetCount()

for i in range(device_count):
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)

        print("========================================")
        print(f"GPU Index: {i}")
        print(f"Name: {pynvml.nvmlDeviceGetName(handle)} || Utilization: {util.gpu}%")

        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print(f"Memory Usage (free/total): {info.free / 1024 / 1024}/{info.total / 1024 / 1024}")
        print("========================================\n")
    except Exception:
        print("GPU ", i, " can't be accessed")

pynvml.nvmlShutdown()