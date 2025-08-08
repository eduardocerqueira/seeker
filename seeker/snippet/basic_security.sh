#date: 2025-08-08T17:12:07Z
#url: https://api.github.com/gists/151b6357dbea21e5a40539db87ea3c3c
#owner: https://api.github.com/users/evandn

#!/bin/bash

# Linux Server Hardening Script for AlmaLinux and Ubuntu
# Author: Van Nguyen
# Version: 1.0

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Global variables
OS_TYPE=""
SSH_PORT=""
NEW_USER=""
USER_PASSWORD= "**********"
LOG_FILE="/var/log/server_hardening.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_FILE
}

# Print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
    log "INFO: $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    log "WARNING: $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    log "ERROR: $1"
}

# Generate random password
generate_password() {
    # Generate 16-character password with mixed case, numbers and symbols
    USER_PASSWORD= "**********"=+/" | cut -c1-16)
    if [[ -z "$USER_PASSWORD" ]]; then
        # Fallback method if openssl fails
        USER_PASSWORD= "**********"
    fi
}

# Detect OS function
detect_os() {
    if [[ -f /etc/redhat-release ]]; then
        OS_TYPE="rhel"
        print_status "Detected RHEL-based system (AlmaLinux/CentOS/RHEL)"
    elif [[ -f /etc/debian_version ]]; then
        OS_TYPE="debian"
        print_status "Detected Debian-based system (Ubuntu/Debian)"
    else
        print_error "Unsupported operating system"
        exit 1
    fi
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root"
        exit 1
    fi
}

# Step 1: Update system
update_system() {
    print_status "Step 1: Updating system packages..."
    
    if [[ $OS_TYPE == "debian" ]]; then
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -y
        apt-get upgrade -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"
        apt-get autoremove -y
        apt-get autoclean
    elif [[ $OS_TYPE == "rhel" ]]; then
        dnf update -y --nobest --skip-broken
        dnf autoremove -y
    fi
    
    print_status "System update completed"
}

# Step 2: Create sudo user and setup SSH keys
setup_user_and_ssh() {
    print_status "Step 2: Setting up user and SSH configuration..."
    
    # Get username from user
    read -p "Enter username for new sudo user: " NEW_USER
    
    # Generate random password for the user
    generate_password
    
    # Create user
    if ! id "$NEW_USER" &>/dev/null; then
        useradd -m -s /bin/bash "$NEW_USER"
        echo "$NEW_USER: "**********"
        usermod -aG sudo "$NEW_USER" 2>/dev/null || usermod -aG wheel "$NEW_USER"
        print_status "Created user: "**********"
    else
        print_warning "User $NEW_USER already exists"
        echo "$NEW_USER: "**********"
        print_status "Updated password for existing user: "**********"
    fi
    
    # Setup SSH directory and generate keys
    USER_HOME="/home/$NEW_USER"
    SSH_DIR="$USER_HOME/.ssh"
    
    mkdir -p "$SSH_DIR"
    chmod 700 "$SSH_DIR"
    
    # Generate SSH key pair
    ssh-keygen -t rsa -b 4096 -f "$SSH_DIR/id_rsa" -N "" -C "$NEW_USER@$(hostname)"
    cp "$SSH_DIR/id_rsa.pub" "$SSH_DIR/authorized_keys"
    chmod 600 "$SSH_DIR/authorized_keys"
    chmod 600 "$SSH_DIR/id_rsa"
    chown -R "$NEW_USER:$NEW_USER" "$SSH_DIR"
    
    print_status "SSH keys generated and configured"
    
    # Configure SSH daemon
    cp /etc/ssh/sshd_config /etc/ssh/sshd_config.backup
    
    # Disable root login and password authentication
    sed -i 's/#*PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
    sed -i 's/#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
    sed -i 's/#*PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
    sed -i 's/#*ChallengeResponseAuthentication.*/ChallengeResponseAuthentication no/' /etc/ssh/sshd_config
    
    print_status "SSH configuration updated"
}

# Step 3: Setup firewall and random SSH port
setup_firewall() {
    print_status "Step 3: Configuring firewall and SSH port..."
    
    # Generate random SSH port between 10000-65000
    SSH_PORT=$(shuf -i 10000-65000 -n 1)
    
    if [[ $OS_TYPE == "debian" ]]; then
        # Install and configure UFW
        apt-get install -y ufw
        ufw --force reset
        ufw default deny incoming
        ufw default allow outgoing
        ufw allow $SSH_PORT/tcp
        ufw allow 80/tcp
        ufw allow 443/tcp
        ufw --force enable
        print_status "UFW configured with SSH port $SSH_PORT"
    elif [[ $OS_TYPE == "rhel" ]]; then
        # Install and configure firewalld
        dnf install -y firewalld
        systemctl enable firewalld
        systemctl start firewalld
        firewall-cmd --permanent --remove-service=ssh
        firewall-cmd --permanent --add-port=$SSH_PORT/tcp
        firewall-cmd --permanent --add-service=http
        firewall-cmd --permanent --add-service=https
        firewall-cmd --reload
        print_status "Firewalld configured with SSH port $SSH_PORT"
    fi
    
    # Update SSH port in configuration
    sed -i "s/#*Port.*/Port $SSH_PORT/" /etc/ssh/sshd_config
    
    # Handle SELinux for SSH port change
    if command -v semanage &> /dev/null; then
        semanage port -a -t ssh_port_t -p tcp $SSH_PORT 2>/dev/null || \
        semanage port -m -t ssh_port_t -p tcp $SSH_PORT
        print_status "SELinux policy updated for SSH port $SSH_PORT"
    fi
}

# Step 4: Setup swap if not exists
setup_swap() {
    print_status "Step 4: Checking and configuring swap..."
    
    if [[ $(swapon --show | wc -l) -eq 0 ]]; then
        # Calculate swap size (half of RAM, max 10GB)
        TOTAL_RAM=$(free -m | awk '/^Mem:/{print $2}')
        SWAP_SIZE=$(($TOTAL_RAM / 2))
        
        if [[ $SWAP_SIZE -gt 10240 ]]; then
            SWAP_SIZE=10240
        fi
        
        print_status "Creating ${SWAP_SIZE}MB swap file..."
        
        fallocate -l ${SWAP_SIZE}M /swapfile
        chmod 600 /swapfile
        mkswap /swapfile
        swapon /swapfile
        
        # Add to fstab for persistence
        if ! grep -q "/swapfile" /etc/fstab; then
            echo '/swapfile none swap sw 0 0' >> /etc/fstab
        fi
        
        print_status "Swap configured successfully"
    else
        print_status "Swap already configured"
    fi
}

# Step 5: Install and configure fail2ban
setup_fail2ban() {
    print_status "Step 5: Installing and configuring fail2ban..."
    
    if [[ $OS_TYPE == "debian" ]]; then
        apt-get install -y fail2ban
    elif [[ $OS_TYPE == "rhel" ]]; then
        dnf install -y epel-release
        dnf install -y fail2ban
    fi
    
    # Configure fail2ban for SSH
    cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 10

[sshd]
enabled = true
port = $SSH_PORT
filter = sshd
logpath = /var/log/auth.log
maxretry = 10
bantime = 3600
EOF
    
    # For RHEL-based systems, adjust log path
    if [[ $OS_TYPE == "rhel" ]]; then
        sed -i 's|/var/log/auth.log|/var/log/secure|' /etc/fail2ban/jail.local
    fi
    
    systemctl enable fail2ban
    systemctl restart fail2ban
    
    print_status "Fail2ban configured and started"
}

# Step 6: Display summary and reboot option
display_summary() {
    print_status "Step 6: Hardening completed! Here's the summary:"
    
    echo -e "\n${BLUE}=== SERVER HARDENING SUMMARY ===${NC}"
    echo -e "OS Type: ${GREEN}$([[ $OS_TYPE == "debian" ]] && echo "Ubuntu/Debian" || echo "AlmaLinux/RHEL")${NC}"
    echo -e "New SSH Port: ${GREEN}$SSH_PORT${NC}"
    echo -e "New User: ${GREEN}$NEW_USER${NC}"
    echo -e "User Password: "**********"
    echo -e "SSH Key Location: ${GREEN}/home/$NEW_USER/.ssh/id_rsa${NC}"
    echo -e "Firewall: ${GREEN}$([[ $OS_TYPE == "debian" ]] && echo "UFW" || echo "Firewalld") - Active${NC}"
    echo -e "Fail2ban: ${GREEN}Active (1h ban, 10 max retries)${NC}"
    echo -e "Swap: ${GREEN}$(free -h | awk '/^Swap:/{print $2}') configured${NC}"
    echo -e "Root SSH: ${RED}Disabled${NC}"
    echo -e "Password Auth: "**********"
    
    echo -e "\n${YELLOW}IMPORTANT CREDENTIALS:${NC}"
    echo -e "Username: ${GREEN}$NEW_USER${NC}"
    echo -e "Password: "**********"
    echo -e "SSH Port: ${GREEN}$SSH_PORT${NC}"
    echo -e "Private Key: ${GREEN}/home/$NEW_USER/.ssh/id_rsa${NC}"
    
    echo -e "\n${YELLOW}CONNECTION METHODS:${NC}"
    echo -e "1. SSH with key: ${GREEN}ssh -i /path/to/key -p $SSH_PORT $NEW_USER@your_server_ip${NC}"
    echo -e "2. Console login: "**********": $NEW_USER, Password: "**********"${NC}"
    
    echo -e "\n${RED}SECURITY NOTES:${NC}"
    echo -e "• SSH password authentication is DISABLED"
    echo -e "• Root SSH login is DISABLED"
    echo -e "• Copy the private key before closing this session!"
    echo -e "• Save these credentials in a secure location"
    echo -e "• Test SSH connection before closing current session!"
    
    # Save credentials to a file for reference
    cat > /root/server_credentials.txt << EOF
=== SERVER HARDENING CREDENTIALS ===
Date: $(date)
OS Type: $([[ $OS_TYPE == "debian" ]] && echo "Ubuntu/Debian" || echo "AlmaLinux/RHEL")
New SSH Port: $SSH_PORT
Username: $NEW_USER
Password: "**********"
SSH Key Location: /home/$NEW_USER/.ssh/id_rsa

Connection Commands:
SSH with key: ssh -i /path/to/key -p $SSH_PORT $NEW_USER@your_server_ip
Console login: Username: $NEW_USER, Password: "**********"

IMPORTANT: 
- Copy the private key from /home/$NEW_USER/.ssh/id_rsa
- SSH password authentication is DISABLED
- Root SSH login is DISABLED
EOF
    
    print_status "Credentials saved to: ${GREEN}/root/server_credentials.txt${NC}"
    
    # Ask for reboot
    echo -e "\n"
    read -p "Do you want to reboot the system now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Rebooting system in 10 seconds..."
        sleep 10
        reboot
    else
        print_status "Please reboot manually when convenient"
        print_warning "Remember to restart SSH service: systemctl restart sshd"
    fi
}

# Main function
main() {
    clear
    echo -e "${BLUE}=== Linux Server Hardening Script ===${NC}"
    echo -e "${BLUE}=== AlmaLinux & Ubuntu Compatible ===${NC}\n"
    
    check_root
    detect_os
    
    print_status "Starting server hardening process..."
    
    update_system
    setup_user_and_ssh
    setup_firewall
    setup_swap
    setup_fail2ban
    
    # Restart SSH service
    systemctl restart sshd
    
    display_summary
}

# Run main function
main "$@"
 setup_firewall
    setup_swap
    setup_fail2ban
    
    # Restart SSH service
    systemctl restart sshd
    
    display_summary
}

# Run main function
main "$@"
