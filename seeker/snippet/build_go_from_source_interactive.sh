#date: 2026-01-16T17:16:34Z
#url: https://api.github.com/gists/40211399890d80d40997dcb38b2fb37e
#owner: https://api.github.com/users/cyberofficial

#!/bin/bash

# Script to download, build, and install Go from source (interactive version)
# Designed for ARM64 (aarch64) Debian 12 in proot environment

set -e  # Exit on error

# Configuration
BUILD_DIR="/tmp/go-build"
BOOTSTRAP_DIR="/tmp/go-bootstrap"
INSTALL_DIR="/usr/local/go"

# Function to check if Go is installed
check_go_installed() {
    if command -v go &> /dev/null; then
        INSTALLED_VERSION=$(go version 2>/dev/null | grep -oP 'go\d+\.\d+\.\d+' | sed 's/^go//')
        INSTALLED_LOCATION=$(which go 2>/dev/null)
        INSTALLED_GOROOT=$(go env GOROOT 2>/dev/null)
        return 0
    else
        return 1
    fi
}

# Function to uninstall Go
uninstall_go() {
    echo "Uninstalling Go..."

    # Remove Go installation
    if [ -d "${INSTALL_DIR}" ]; then
        echo "Removing Go from ${INSTALL_DIR}..."
        sudo rm -rf "${INSTALL_DIR}"
    fi

    # Remove Go from PATH configuration
    echo "Removing Go from system profiles..."

    # Remove from /etc/profile
    if [ -f /etc/profile ]; then
        sudo sed -i '/# Go .* from source/,/export PATH=\$PATH:'"${INSTALL_DIR}"'/bin/d' /etc/profile 2>/dev/null || true
    fi

    # Remove from /etc/bash.bashrc
    if [ -f /etc/bash.bashrc ]; then
        sudo sed -i '/# Go .* from source/,/export PATH=\$PATH:'"${INSTALL_DIR}"'/bin/d' /etc/bash.bashrc 2>/dev/null || true
    fi

    echo "✓ Go has been uninstalled"
    echo ""
}

# Check if Go is already installed
echo "=========================================="
echo "Go Source Build Script (Interactive)"
echo "=========================================="
echo ""

if check_go_installed; then
    echo "Go is already installed!"
    echo ""
    echo "  Version:    ${INSTALLED_VERSION}"
    echo "  Location:   ${INSTALLED_LOCATION}"
    echo "  GOROOT:     ${INSTALLED_GOROOT}"
    echo ""
    echo "What would you like to do?"
    echo "  1) Reinstall current version (${INSTALLED_VERSION})"
    echo "  2) Install different version (overwrite existing)"
    echo "  3) Remove/Uninstall Go completely"
    echo "  4) Cancel and exit"
    echo ""
    read -p "Enter choice [1-4]: " INSTALL_CHOICE

    case $INSTALL_CHOICE in
        1)
            echo ""
            echo "Reinstalling Go ${INSTALLED_VERSION}..."
            GO_VERSION="${INSTALLED_VERSION}"
            ;;
        2)
            echo ""
            # Continue to version selection below
            ;;
        3)
            echo ""
            uninstall_go
            echo "Uninstallation complete. Exiting..."
            exit 0
            ;;
        4)
            echo ""
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo ""
            echo "Invalid choice. Exiting..."
            exit 1
            ;;
    esac
else
    echo "Go is not currently installed."
    echo ""
fi

# Function to fetch latest Go version
get_latest_go_version() {
    LATEST_VERSION=$(curl -s https://go.dev/VERSION?m=text | head -n 1)
    # Remove "go" prefix to get just the version number
    echo "${LATEST_VERSION#go}"
}

# Ask user for version choice
echo "Select Go version to install:"
echo "  1) Latest stable version"
echo "  2) Custom version"
echo ""
read -p "Enter choice [1-2]: " VERSION_CHOICE

if [ "$VERSION_CHOICE" = "1" ]; then
    echo ""
    echo "Fetching latest Go version..."
    GO_VERSION=$(get_latest_go_version)
    echo ""
    echo "Latest Go version detected: $GO_VERSION"
elif [ "$VERSION_CHOICE" = "2" ]; then
    read -p "Enter Go version (e.g., 1.25.6): " GO_VERSION
else
    echo "Invalid choice. Using latest version..."
    GO_VERSION=$(get_latest_go_version)
fi

# Validate version format
if ! [[ $GO_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Invalid version format. Expected format: X.Y.Z (e.g., 1.25.6)"
    exit 1
fi

echo ""
echo "Selected Go version: $GO_VERSION"
echo ""

# Confirm before proceeding
read -p "Proceed with installing Go $GO_VERSION? [y/N]: " CONFIRM_PROCEED
if [[ ! "$CONFIRM_PROCEED" =~ ^[Yy]$ ]]; then
    echo "Installation cancelled by user."
    exit 0
fi

echo ""

# Set bootstrap version based on target version
# For Go 1.22+, we need a bootstrap version >= 1.20
MAJOR_VERSION=$(echo "$GO_VERSION" | cut -d. -f1)
MINOR_VERSION=$(echo "$GO_VERSION" | cut -d. -f2)

if [ "$MAJOR_VERSION" -eq 1 ] && [ "$MINOR_VERSION" -ge 22 ]; then
    BOOTSTRAP_VERSION="1.23.6"
elif [ "$MAJOR_VERSION" -eq 1 ] && [ "$MINOR_VERSION" -ge 20 ]; then
    BOOTSTRAP_VERSION="1.21.6"
else
    BOOTSTRAP_VERSION="1.19.13"
fi

GO_SOURCE_URL="https://go.dev/dl/go${GO_VERSION}.src.tar.gz"
BOOTSTRAP_URL="https://go.dev/dl/go${BOOTSTRAP_VERSION}.linux-arm64.tar.gz"

echo "Bootstrap Go version: $BOOTSTRAP_VERSION"
echo ""


# Phase 1: Install dependencies and download bootstrap Go
echo "[1/6] Installing build dependencies and downloading bootstrap Go..."
sudo apt update
sudo apt install -y bison

# Download bootstrap Go binary
mkdir -p "${BOOTSTRAP_DIR}"
cd "${BOOTSTRAP_DIR}"

if command -v curl &> /dev/null; then
    echo "Using curl to download bootstrap Go ${BOOTSTRAP_VERSION}..."
    curl -L -o "go-bootstrap.tar.gz" "${BOOTSTRAP_URL}"
elif command -v wget &> /dev/null; then
    echo "Using wget to download bootstrap Go ${BOOTSTRAP_VERSION}..."
    wget -O "go-bootstrap.tar.gz" "${BOOTSTRAP_URL}"
fi

# Extract bootstrap Go
tar -xzf "go-bootstrap.tar.gz"
echo "✓ Dependencies installed and bootstrap Go downloaded"
echo ""

# Phase 2: Download Go source
echo "[2/6] Downloading Go source..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Try curl first, fallback to wget
if command -v curl &> /dev/null; then
    echo "Using curl to download..."
    curl -L -o "go${GO_VERSION}.src.tar.gz" "${GO_SOURCE_URL}"
elif command -v wget &> /dev/null; then
    echo "Using wget to download..."
    wget -O "go${GO_VERSION}.src.tar.gz" "${GO_SOURCE_URL}"
else
    echo "ERROR: Neither curl nor wget found!"
    exit 1
fi

# Verify download
if [ ! -f "go${GO_VERSION}.src.tar.gz" ]; then
    echo "ERROR: Failed to download Go source!"
    exit 1
fi

echo "✓ Go source downloaded"
echo ""

# Phase 3: Extract and build
echo "[3/6] Extracting Go source..."
tar -xzf "go${GO_VERSION}.src.tar.gz"
cd "go"

echo "[4/6] Building Go ${GO_VERSION}..."
echo "This may take 15-30 minutes on ARM64..."
echo ""

# Set bootstrap compiler
export GOROOT_BOOTSTRAP="${BOOTSTRAP_DIR}/go"

# Build Go from src directory
cd src
./make.bash
cd ..

echo "✓ Go build completed"
echo ""

# Phase 5: Install to system
echo "[5/6] Installing Go to ${INSTALL_DIR}..."
sudo rm -rf "${INSTALL_DIR}"
sudo mkdir -p "${INSTALL_DIR}"
sudo mv * "${INSTALL_DIR}/"

echo "✓ Go installed"
echo ""

# Phase 6: Cleanup and configuration
echo "[6/6] Cleaning up and configuring..."

# Clean up bootstrap Go
echo "Cleaning up bootstrap Go..."
rm -rf "${BOOTSTRAP_DIR}"

# Clean up build directory
echo "Cleaning temporary files..."
cd /tmp
rm -rf "${BUILD_DIR}"

# Configure PATH in system profile
if ! grep -q "${INSTALL_DIR}/bin" /etc/profile 2>/dev/null; then
    echo "" | sudo tee -a /etc/profile > /dev/null
    echo "# Go ${GO_VERSION} from source" | sudo tee -a /etc/profile > /dev/null
    echo "export GOROOT=${INSTALL_DIR}" | sudo tee -a /etc/profile > /dev/null
    echo "export PATH=\$PATH:${INSTALL_DIR}/bin" | sudo tee -a /etc/profile > /dev/null
fi

# Also add to bash.bashrc for immediate effect in new shells
if ! grep -q "${INSTALL_DIR}/bin" /etc/bash.bashrc 2>/dev/null; then
    echo "" | sudo tee -a /etc/bash.bashrc > /dev/null
    echo "# Go ${GO_VERSION} from source" | sudo tee -a /etc/bash.bashrc > /dev/null
    echo "export GOROOT=${INSTALL_DIR}" | sudo tee -a /etc/bash.bashrc > /dev/null
    echo "export PATH=\$PATH:${INSTALL_DIR}/bin" | sudo tee -a /etc/bash.bashrc > /dev/null
fi

echo "✓ Cleanup completed"
echo ""

# Set up environment for current session
export GOROOT=${INSTALL_DIR}
export PATH=$PATH:${INSTALL_DIR}/bin

# Verify installation
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Verifying installation..."
${INSTALL_DIR}/bin/go version
echo ""
echo "Go ${GO_VERSION} has been installed to: ${INSTALL_DIR}"
echo ""
echo "To use Go in your current shell, run:"
echo "  export GOROOT=${INSTALL_DIR}"
echo "  export PATH=\$PATH:${INSTALL_DIR}/bin"
echo ""
echo "Or simply log out and log back in for the changes to take effect."
echo ""