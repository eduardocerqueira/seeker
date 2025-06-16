#date: 2025-06-16T16:54:39Z
#url: https://api.github.com/gists/f3c9774ed8d19643fd818ed47c75933a
#owner: https://api.github.com/users/luizcosta

#!/bin/bash

# WSL SSH Connection Setup Script for macOS
# This script helps configure SSH connection to WSL Ubuntu for Cursor remote development

echo "=== WSL SSH Connection Setup for Cursor ==="
echo

# Check if SSH key exists
if [ ! -f ~/.ssh/id_ed25519 ]; then
    echo "No SSH key found. Generating a new one..."
    ssh-keygen -t ed25519 -C "cursor-wsl-connection"
    echo
fi

# Get WSL connection details
read -p "Enter your Windows machine IP address: " WINDOWS_IP
read -p "Enter your WSL username: " WSL_USER
read -p "Enter SSH port (default 2222): " SSH_PORT
SSH_PORT=${SSH_PORT:-2222}

# Display public key
echo
echo "Your SSH public key (copy this to WSL):"
echo "======================================="
cat ~/.ssh/id_ed25519.pub
echo "======================================="
echo

# Test connection
echo "Testing SSH connection..."
echo "Command: ssh -p $SSH_PORT $WSL_USER@$WINDOWS_IP"
echo

read -p "Do you want to add this connection to your SSH config? (y/n): " ADD_CONFIG

if [ "$ADD_CONFIG" = "y" ]; then
    # Backup existing config
    if [ -f ~/.ssh/config ]; then
        cp ~/.ssh/config ~/.ssh/config.backup
        echo "Backed up existing SSH config to ~/.ssh/config.backup"
    fi
    
    # Add WSL host to SSH config
    echo "
Host wsl-ubuntu
    HostName $WINDOWS_IP
    Port $SSH_PORT
    User $WSL_USER
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3" >> ~/.ssh/config
    
    echo "Added 'wsl-ubuntu' to your SSH config"
    echo "You can now connect using: ssh wsl-ubuntu"
    echo
fi

# Instructions for Cursor
echo "=== Cursor Setup Instructions ==="
echo "1. Open Cursor"
echo "2. Open Command Palette (Cmd+Shift+P)"
echo "3. Type 'Remote-SSH: Add New SSH Host'"
echo "4. Enter: ssh -p $SSH_PORT $WSL_USER@$WINDOWS_IP"
echo "   Or if you added to config: ssh wsl-ubuntu"
echo "5. Select your SSH config file location"
echo "6. Then use 'Remote-SSH: Connect to Host' to connect"
echo

echo "=== Next Steps on WSL Ubuntu ==="
echo "1. Install OpenSSH server: sudo apt install openssh-server"
echo "2. Configure SSH port in /etc/ssh/sshd_config to $SSH_PORT"
echo "3. Add your public key to ~/.ssh/authorized_keys"
echo "4. Restart SSH: sudo systemctl restart ssh"
echo

echo "For detailed instructions, see: ~/wsl-ssh-setup.md" 