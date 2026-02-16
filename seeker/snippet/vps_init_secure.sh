#date: 2026-02-16T17:23:48Z
#url: https://api.github.com/gists/28b8831648993bceeed023f480459c27
#owner: https://api.github.com/users/evg-bot

#!/bin/bash
set -e

if [ "$EUID" -ne 0 ]; then
  echo "Запусти от root (sudo)"
  exit 1
fi


SSH_PORT=22
SWAP_SIZE=1G

apt update
apt upgrade -y
apt autoremove -y

cat > /etc/sysctl.d/99-disable-ipv6.conf <<EOF
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1
EOF
sysctl --system

apt install ufw -y
ufw default deny incoming
ufw default allow outgoing
ufw allow ${SSH_PORT}/tcp
ufw --force enable

apt install fail2ban -y
mkdir -p /etc/fail2ban/jail.d

cat > /etc/fail2ban/jail.d/sshd.local <<EOF
[sshd]
enabled = true
port = ${SSH_PORT}
backend = systemd
maxretry = 5
findtime = 10m
bantime = 1h
EOF

systemctl enable fail2ban
systemctl restart fail2ban

if ! swapon --show | grep -q swapfile; then
    fallocate -l ${SWAP_SIZE} /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

ufw status verbose
fail2ban-client status
swapon --show
