#date: 2025-02-19T16:33:39Z
#url: https://api.github.com/gists/5b67ebb1d0d5fcae83c5ad6c1565a9e9
#owner: https://api.github.com/users/HallexCosta

#!/bin/bash

# Update package lists
echo "Updating package lists..."
sudo apt update

# Install build-essential, kernel headers, and dkms
echo "Installing build-essential, kernel headers, and dkms..."
sudo apt install -y build-essential linux-headers-$(uname -r) dkms

# Download VMware Workstation Pro bundle
echo "Downloading VMware Workstation Pro 17.5..."
# don´t work (you need put your url for download)
#wget -O vmware-workstation.bundle https://download3.vmware.com/software/WKST-1750-LX/VMware-Workstation-Full-17.5.0-22583795.x86_64.bundle

# Make the VMware bundle executable
echo "Making VMware Workstation Pro bundle executable..."
chmod +x vmware-workstation.bundle

echo "Installation script has finished."
sudo ./vmware-workstation.bundle