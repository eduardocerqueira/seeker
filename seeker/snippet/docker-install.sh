#date: 2026-01-19T17:11:43Z
#url: https://api.github.com/gists/c43bacb372b97607f67fd585947ea86f
#owner: https://api.github.com/users/CovertCode

#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Docker & Portainer Installation..."

# 1. Install Prerequisites
echo "Installing prerequisites..."
sudo apt update
sudo apt install -y ca-certificates curl gnupg

# 2. Add Docker's Official GPG Key
echo "Adding Docker GPG key..."
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# 3. Add the Repository to Apt Sources
echo "Adding Docker repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 4. Install Docker Packages
echo "Installing Docker Engine..."
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 5. Post-Installation Steps (Run without sudo)
echo "Adding current user ($USER) to the docker group..."
sudo usermod -aG docker $USER

# 6. Install Portainer (Latest)
echo "Installing Portainer..."
# Ensure Docker service is started
sudo systemctl start docker

# Pull and Run Portainer
# Note: We use 'sudo' here because the group change above won't apply until you log out
sudo docker pull portainer/portainer-ce:latest
sudo docker run -d \
  -p 8000:8000 \
  -p 9443:9443 \
  -p 9000:9000 \
  --name=portainer \
  --restart=always \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce:latest

echo "----------------------------------------------------"
echo "Installation Complete!"
echo "1. Portainer is running at: https://localhost:9443"
echo "2. IMPORTANT: You must LOG OUT and LOG BACK IN for permission changes to take effect (to run 'docker' without 'sudo')."
echo "----------------------------------------------------"