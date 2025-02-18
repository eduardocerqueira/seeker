#date: 2025-02-18T17:03:42Z
#url: https://api.github.com/gists/a9b969a0102847720d57a6fa3a69f426
#owner: https://api.github.com/users/Atharva-3000

#!/bin/bash
set -e

echo "Installing custom fingerprint driver for VID: 2808 and PID: a658 on Fedora"

# Create temporary working directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Download the rpm package
echo "Downloading the custom driver package..."
wget -q https://github.com/ftfpteams/RTS5811-FT9366-fingerprint-linux-driver-with-VID-2808-and-PID-a658/raw/main/libfprint-2-2-1.94.4+tod1-FT9366_20240627.x86_64.rpm

# Extract the rpm without installing it (to avoid package conflicts)
echo "Extracting the package contents..."
mkdir -p rpm_extract
cd rpm_extract
rpm2cpio ../libfprint-2-2-1.94.4+tod1-FT9366_20240627.x86_64.rpm | cpio -idmv
cd ..

# Install fprintd and dependencies (BUT NOT libfprint)
echo "Installing fprintd and dependencies..."
sudo dnf install -y fprintd-pam
# Note: We're intentionally not installing the libfprint package because we'll use our custom one

# Copy the custom library to the correct location 
echo "Installing the custom driver library..."
sudo mkdir -p /usr/lib64
sudo cp rpm_extract/usr/lib64/libfprint-2.so.2.0.0 /usr/lib64/

# Create the required symlinks
echo "Creating necessary symlinks..."
sudo ln -sf /usr/lib64/libfprint-2.so.2.0.0 /usr/lib64/libfprint-2.so
sudo ln -sf /usr/lib64/libfprint-2.so.2.0.0 /usr/lib64/libfprint-2.so.2

# Reload the library cache
sudo ldconfig

# Install fprintd (now that our custom library is in place)
echo "Installing fprintd..."
sudo dnf install -y fprintd --setopt=strict=0

# Enable and start the service
echo "Enabling fprintd service..."
sudo systemctl enable --now fprintd.service

# Enable fingerprint authentication
echo "Configuring authentication..."
sudo authselect enable-feature with-fingerprint
sudo authselect apply-changes

# Clean up
cd "$HOME"
rm -rf "$TEMP_DIR"

echo "Installation complete. You can now enroll your fingerprint with 'fprintd-enroll'"