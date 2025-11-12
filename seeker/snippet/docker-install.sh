#date: 2025-11-12T16:58:18Z
#url: https://api.github.com/gists/64d53c66de161a98eb9615f5edb85316
#owner: https://api.github.com/users/Youssef-Lehmam

#!/usr/bin/env bash
set -euo pipefail

USER_TO_ENABLE="ubuntu"

echo "[1/7] Cleaning old Docker versions..."
apt-get remove -y docker docker-engine docker.io containerd runc || true

echo "[2/7] Installing prerequisites..."
apt-get update -y
apt-get install -y ca-certificates curl gnupg lsb-release apt-transport-https

echo "[3/7] Adding Docker GPG key..."
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo "[4/7] Adding Docker APT repository..."
. /etc/os-release
ARCH=$(dpkg --print-architecture)
echo \
"deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
${VERSION_CODENAME} stable" > /etc/apt/sources.list.d/docker.list

echo "[5/7] Installing Docker Engine + Compose..."
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "[6/7] Enabling Docker and allowing non-sudo usage..."
systemctl enable --now docker
groupadd -f docker
usermod -aG docker "$USER_TO_ENABLE"

# Optional immediate socket access (avoids relog)
if command -v setfacl >/dev/null 2>&1; then
  setfacl -m "u:${USER_TO_ENABLE}:rw" /var/run/docker.sock || true
fi

echo "[7/7] Checking versions..."
docker --version
docker compose version
echo "✅ Docker installation complete."
echo "ℹ️  If 'permission denied', log out/in or run: newgrp docker"