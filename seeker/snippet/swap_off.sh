#date: 2023-06-26T16:59:37Z
#url: https://api.github.com/gists/52a4642fc31a7e110fe1145608b0e3df
#owner: https://api.github.com/users/pythoninthegrass

#!/usr/bin/env bash

# To see current swap usage
sysctl -a | grep swap

# Monitor swap usage
vm_stat 60

# Use only when when your system is in a very bad shape
sudo pkill -HUP -u _windowserver

# To monitor, what's creating/updating these swap files
sudo fs_usage | grep swapfile

# Or for page ins/outs
sudo fs_usage | grep PAGE_

# To see what WindowServer process is doing exactly
sudo spindump -reveal $(pgrep WindowServer)

# or for kernel_task
sudo spindump -reveal 0
