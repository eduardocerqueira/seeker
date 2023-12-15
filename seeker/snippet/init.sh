#date: 2023-12-15T16:53:44Z
#url: https://api.github.com/gists/eb22076cff3bc50a5a4d3e348dd46212
#owner: https://api.github.com/users/zsnmwy

REMOTE=https://gitee.com/mirrors/oh-my-zsh.git sh -c "$(curl -fsSL https://gitee.com/mirrors/oh-my-zsh/raw/master/tools/install.sh)"  "" --unattended
sed -i  s/robbyrussell/ys/ ~/.zshrc

chsh -s $(which zsh)

kubectl completion zsh > "/root/.oh-my-zsh/plugins/git/_kubectl"
helm completion zsh > "/root/.oh-my-zsh/plugins/git/_helm"

IP=$(ip -j -p a | jq  '.[].addr_info | .[] | select(.label == "ens192") | .local' | cut -d '"' -f2)

hostnamectl set-hostname "ubuntu-$(echo $IP | tr '.' '-')"

modprobe overlay
modprobe br_netfilter
sysctl --system

cat <<EOF | tee /etc/netplan/50-cloud-init.yaml
# This file is generated from information provided by the datasource.  Changes
# to it will not persist across an instance reboot.  To disable cloud-init's
# network configuration capabilities, write a file
# /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg with the following:
# network: {config: disabled}
network:
    ethernets:
        ens192:
            addresses:
                - ${IP}/24
            routes:
                - to: 0.0.0.0/0
                  via: 192.168.31.1
            nameservers:
                addresses:
                - 192.168.31.1
    version: 2
EOF

cat <<EOF | tee /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg
network: {config: disabled}
EOF

netplan apply
