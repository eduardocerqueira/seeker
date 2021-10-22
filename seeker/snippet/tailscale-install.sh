#date: 2021-10-22T16:56:41Z
#url: https://api.github.com/gists/1f765e21a05b6542787fa0714ce04072
#owner: https://api.github.com/users/graytonio

#!/bin/bash

curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/focal.gpg | sudo apt-key add -
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/focal.list | sudo tee /etc/apt/sources.list.d/tailscale.list

sudo apt update
sudo apt install tailscale

sudo tailscale up
tailscale ip -4