#date: 2025-03-04T17:05:07Z
#url: https://api.github.com/gists/d03f65300f69520be672e8d8b6d44e54
#owner: https://api.github.com/users/NonsoAmadi10

#!/bin/bash

# Exit on error, undefined variable usage, and prevent pipeline failures
set -euo pipefail

# Function to print messages
print_message() {
    echo -e "\n\033[1;34m$1\033[0m"  # Blue color for info
}

print_success() {
    echo -e "\n\033[1;32m✅ $1\033[0m"  # Green color for success
}

print_error() {
    echo -e "\n\033[1;31m❌ $1\033[0m"  # Red color for errors
    exit 1
}

# Update system
print_message "🔄 Updating system packages..."
sudo apt update && sudo apt upgrade -y || print_error "Failed to update system packages."

# Install dependencies
print_message "📦 Installing dependencies (wget, gpg)..."
sudo apt install -y wget gpg || print_error "Failed to install dependencies."

# Set Bitcoin Core version
BITCOIN_VERSION="26.0"
BITCOIN_TAR="bitcoin-${BITCOIN_VERSION}-x86_64-linux-gnu.tar.gz"
BITCOIN_DIR="/opt/bitcoin-${BITCOIN_VERSION}"
BITCOIN_URL="https://bitcoincore.org/bin/bitcoin-core-${BITCOIN_VERSION}/${BITCOIN_TAR}"

# Download Bitcoin Core
print_message "⬇️ Downloading Bitcoin Core v${BITCOIN_VERSION}..."
wget -q --show-progress "$BITCOIN_URL" || print_error "Failed to download Bitcoin Core."

# Verify checksum
print_message "🔍 Verifying download integrity..."
wget -q https://bitcoincore.org/bin/bitcoin-core-${BITCOIN_VERSION}/SHA256SUMS || print_error "Failed to download SHA256SUMS."
wget -q https://bitcoincore.org/bin/bitcoin-core-${BITCOIN_VERSION}/SHA256SUMS.asc || print_error "Failed to download SHA256SUMS.asc."

# Import Bitcoin Core signing key
gpg --keyserver hkps://keyserver.ubuntu.com --recv-keys 01EA5486DE18A882D4C2684590C8019E36C2E964 || print_error "Failed to import GPG key."

# Verify signature
gpg --verify SHA256SUMS.asc SHA256SUMS || print_error "GPG signature verification failed."

# Check file integrity
grep "$BITCOIN_TAR" SHA256SUMS | sha256sum -c - || print_error "Checksum mismatch. File may be corrupted."

# Extract Bitcoin Core
print_message "📂 Extracting Bitcoin Core..."
sudo tar -xzf "$BITCOIN_TAR" -C /opt || print_error "Failed to extract Bitcoin Core."

# Create Bitcoin directory
print_message "📁 Creating Bitcoin data directory..."
mkdir -p ~/.bitcoin || print_error "Failed to create Bitcoin data directory."

# Secure Bitcoin configuration
print_message "⚙️ Creating Bitcoin configuration file..."
RPC_PASSWORD= "**********"
cat << EOF > ~/.bitcoin/bitcoin.conf
signet=1
[signet]
rpcuser=bitcoinrpc
rpcpassword= "**********"
EOF
chmod 600 ~/.bitcoin/bitcoin.conf  # Restrict permissions for security

# Add Bitcoin binaries to PATH
print_message "🛠️ Adding Bitcoin binaries to PATH..."
{
    echo "export PATH=\$PATH:${BITCOIN_DIR}/bin"
    echo "alias bitcoind='${BITCOIN_DIR}/bin/bitcoind'"
    echo "alias bitcoin-cli='${BITCOIN_DIR}/bin/bitcoin-cli'"
} >> ~/.bashrc

# Source updated profile safely
print_message "🔄 Reloading shell configuration..."
if source ~/.bashrc; then
    print_success "PATH updated successfully."
else
    print_message "⚠️ Warning: Unable to reload shell. Restart your terminal for changes to take effect."
fi

# Clean up downloaded files
print_message "🧹 Cleaning up temporary files..."
rm -f "$BITCOIN_TAR" SHA256SUMS SHA256SUMS.asc || print_error "Failed to clean up files."

# Start Bitcoin daemon
print_message "🚀 Starting Bitcoin daemon..."
bitcoind -daemon || print_error "Failed to start Bitcoin daemon."

# Wait for Bitcoin daemon to start
print_message "⏳ Waiting for Bitcoin daemon to initialize..."
timeout=300
while [ ! -d ~/.bitcoin/signet/blocks ] && [ $timeout -gt 0 ]; do
    sleep 5  # Reduce CPU load by checking every 5 seconds
    ((timeout-=5))
done

if [ $timeout -le 0 ]; then
    print_error "Timeout waiting for Bitcoin daemon to start."
fi

# Check Bitcoin daemon status
print_message "🔍 Checking Bitcoin daemon status..."
bitcoin-cli -signet getblockchaininfo || print_error "Failed to check Bitcoin daemon status."

# Success message
print_success "🎉 Bitcoin Core installation complete and daemon started! Initial block download is in progress."
print_success "You can use the following commands:"
print_success "  - Start Bitcoin daemon: bitcoind -daemon"
print_success "  - Stop Bitcoin daemon: bitcoin-cli -signet stop"
print_success "  - Check blockchain info: bitcoin-cli -signet getblockchaininfo"
kchaininfo"
