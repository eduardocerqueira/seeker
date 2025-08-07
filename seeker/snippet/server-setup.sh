#date: 2025-08-07T17:14:03Z
#url: https://api.github.com/gists/58e92dd5c7541975670e58e796800b9f
#owner: https://api.github.com/users/sohelrana820

#!/bin/bash

set -e

echo "🟡 Updating system (waiting if locked)..."

# Wait if dpkg is locked by unattended-upgrades
while sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1; do
    echo "⚠️  Waiting for apt lock to be released..."
    sleep 5
done

sudo apt update && sudo apt upgrade -y

echo "🔧 Installing Git..."
sudo apt install -y git

echo "🌐 Installing NGINX..."
sudo apt install -y nginx
sudo systemctl enable nginx
sudo systemctl start nginx

echo "🐬 Installing MySQL 8.0..."
sudo apt install -y gnupg wget
wget https://dev.mysql.com/get/mysql-apt-config_0.8.24-1_all.deb
sudo dpkg -i mysql-apt-config_0.8.24-1_all.deb
sudo apt update
sudo DEBIAN_FRONTEND=noninteractive apt install -y mysql-server
rm -f mysql-apt-config_0.8.24-1_all.deb
sudo systemctl enable mysql
sudo systemctl start mysql

echo "🐳 Installing Docker..."
sudo apt install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    gpg --dearmor | \
    sudo tee /etc/apt/keyrings/docker.gpg > /dev/null

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

echo "➕ Adding user to docker group..."
sudo usermod -aG docker "$USER"

echo "✅ Done installing system packages (logout/login or run: newgrp docker)"