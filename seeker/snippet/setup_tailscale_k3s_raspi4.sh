#date: 2024-10-28T17:03:02Z
#url: https://api.github.com/gists/c39cb7a8bafd6177321f4e28a999c7cd
#owner: https://api.github.com/users/eggplants

#!/usr/bin/env bash

# ===
# > master & agent nodes
# ===
sudo apt update -y && sudo apt upgrade -y

sudo systemctl stop dphys-swapfile
sudo systemctl disable dphys-swapfile
sudo rm -f /var/swap

echo 'NTP=ntp.nict.jp' | sudo tee -a /etc/systemd/timesyncd.conf
sudo systemctl restart systemd-timesyncd

curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

curl https://mise.run | sh
echo 'eval "$($HOME/.local/bin/mise activate bash)"' >>~/.bashrc
eval "$($HOME/.local/bin/mise activate bash)"
mise use --global python@latest

echo "$(</boot/firmware/cmdline.txt) cgroup_memory=1 cgroup_enable=memory" > cmdline.txt
cat cmdline.txt | sudo tee /boot/firmware/cmdline.txt
rm cmdline.txt

sudo reboot

# ===
# > master node
# ===
MASTER_NODE_IP="$(tailscale status | awk 'NR==1{print$1}')"
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="server --flannel-iface tailscale0 --advertise-address $MASTER_NODE_IP --node-ip $MASTER_NODE_IP --node-external-ip $MASTER_NODE_IP" sh -s

# ===
# > agent node(s)
# ===
K3S_MASTER_TOKEN= "**********"
MASTER_NODE_HOSTNAME="..."
MASTER_NODE_IP="$(tailscale status | awk -v m="$MASTER_NODE_HOSTNAME" '$2==m{print$1}')"
AGENT_NODE_IP="$(tailscale status | awk 'NR==1{print$1}')"
curl -sfL https: "**********"://${MASTER_NODE_IP}:6443 --token $K3S_MASTER_TOKEN --flannel-iface tailscale0 --node-ip $AGENT_NODE_IP --node-external-ip $AGENT_NODE_IP" sh -sxternal-ip $AGENT_NODE_IP" sh -s