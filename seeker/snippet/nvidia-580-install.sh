#date: 2025-12-15T17:07:32Z
#url: https://api.github.com/gists/72e952041392d95ccb933ec6c0bb9ecb
#owner: https://api.github.com/users/Arthur2500

#!/usr/bin/env bash
set -e

echo "Starting CUDA Driver installation for Debian"

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or with sudo"
  exit 1
fi

echo "Updating system package lists"
apt update
apt -y upgrade

echo "Installing required dependencies"
apt install -y \
  build-essential \
  dkms \
  linux-headers-$(uname -r) \
  ca-certificates \
  curl

echo "Removing existing NVIDIA and CUDA packages if present"
apt purge -y 'nvidia-*' 'cuda-*' || true
apt autoremove -y

echo "Downloading CUDA keyring package"
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
  -o /tmp/cuda-keyring.deb

echo "Installing CUDA keyring"
dpkg -i /tmp/cuda-keyring.deb

echo "Refreshing APT cache"
apt update

echo "Installing NVIDIA 580 pinning"
apt install -y nvidia-driver-pinning-580

echo "Installing NVIDIA CUDA drivers"
apt install -y cuda-drivers-580

echo "Updating initramfs"
update-initramfs -u

echo "Installation completed successfully"
echo "Please reboot the system"