#date: 2025-05-07T16:54:28Z
#url: https://api.github.com/gists/d71a93a9d4da2248ed5322ba2dd6013e
#owner: https://api.github.com/users/rajvermacas

import os, psutil
process = psutil.Process(os.getpid())
mem_info = process.memory_info()
print(f"RSS: {mem_info.rss / (1024*1024):.2f} MB, VMS: {mem_info.vms / (1024*1024):.2f} MB")