#date: 2025-10-09T16:29:57Z
#url: https://api.github.com/gists/59afc438f9df928c5b137dc71c58fd73
#owner: https://api.github.com/users/LhrSupun

#!/bin/bash

# Download the .NET install script
wget https://dot.net/v1/dotnet-install.sh -O dotnet-install.sh

# Make it executable
chmod +x ./dotnet-install.sh

# Install .NET 8 (change channel to 8.0)
./dotnet-install.sh --channel 8.0

# Set up environment variables
export DOTNET_ROOT=$HOME/.dotnet
export PATH=$PATH:$DOTNET_ROOT:$DOTNET_ROOT/tools

# Make environment variables persistent
SHELL_CONFIG=""
if [ -f "$HOME/.bashrc" ]; then
    SHELL_CONFIG="$HOME/.bashrc"
elif [ -f "$HOME/.bash_profile" ]; then
    SHELL_CONFIG="$HOME/.bash_profile"
elif [ -f "$HOME/.zshrc" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
fi

if [ -n "$SHELL_CONFIG" ]; then
    # Check if DOTNET_ROOT already exists in config
    if ! grep -q "DOTNET_ROOT" "$SHELL_CONFIG"; then
        echo "" >> "$SHELL_CONFIG"
        echo "# .NET Configuration" >> "$SHELL_CONFIG"
        echo "export DOTNET_ROOT=\$HOME/.dotnet" >> "$SHELL_CONFIG"
        echo "export PATH=\$PATH:\$DOTNET_ROOT:\$DOTNET_ROOT/tools" >> "$SHELL_CONFIG"
        echo "Environment variables added to $SHELL_CONFIG"
    else
        echo "DOTNET_ROOT already exists in $SHELL_CONFIG"
    fi
fi

# Verify installation
dotnet --version

echo ""
echo "Installation complete! Please run 'source $SHELL_CONFIG' or restart your terminal."