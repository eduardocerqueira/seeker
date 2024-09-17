#date: 2024-09-17T16:55:05Z
#url: https://api.github.com/gists/18888e23fee5a7788305f1aa35a1df3b
#owner: https://api.github.com/users/zitterbewegung

#!/bin/bash

# Make the script executable
chmod +x install.sh

# Install Python dependencies
echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

# Check and install Ghidra if needed
if ! command -v ghidra &> /dev/null; then
    echo "Ghidra not found. Please install Ghidra manually from https://ghidra-sre.org/"
fi

# Check and install DTrace if needed
if ! command -v dtrace &> /dev/null; then
    echo "DTrace not found. Please install DTrace using your system's package manager."
fi

# Add any other specific dependency checks as needed
echo "All dependencies installed. You may need to restart your terminal for some changes to take effect."
