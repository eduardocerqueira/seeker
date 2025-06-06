#date: 2025-06-06T17:11:31Z
#url: https://api.github.com/gists/e3b3cf8a4c5a3f89dec981dc414a5b1f
#owner: https://api.github.com/users/LouisCastricato

#!/bin/bash
#
# Ngrok Setup for Existing SSH Jump Host
# This script adds ngrok to your already-configured secure jump host
# Assumes previous jumphost setup script has been run
#
# Run with: "**********"
#

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Script should NOT be run as root since we need user context
if [[ $EUID -eq 0 ]]; then
   error "This script should NOT be run as root. Run as your regular user."
   exit 1
fi

log "Starting Ngrok Add-on Setup for SSH Jump Host"

# First, we need to temporarily disable ForceCommand for SSH to work with ngrok
log "Adjusting SSH configuration for ngrok compatibility..."
if sudo grep -q "ForceCommand" /etc/ssh/sshd_config.d/99-jumphost.conf; then
    warning "Detected ForceCommand in SSH config - creating exemption for ngrok connections"
    # Comment out the ForceCommand line
    sudo sed -i 's/^ForceCommand/#ForceCommand/' /etc/ssh/sshd_config.d/99-jumphost.conf
    
    # Create a new config that allows both normal SSH and logged sessions
    sudo tee /etc/ssh/sshd_config.d/98-ngrok-compat.conf > /dev/null << 'EOF'
# Allow normal SSH for ngrok connections while maintaining logging for direct connections
Match User !root
    # Logging will be handled by audit and auth.log instead of ForceCommand
EOF
    
    sudo systemctl reload ssh
fi

# Install dependencies
log "Installing dependencies..."
sudo apt update
sudo apt install -y jq curl wget

# Install ngrok using apt repository
log "Installing ngrok via apt..."
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
    | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null

echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
    | sudo tee /etc/apt/sources.list.d/ngrok.list

sudo apt update
sudo apt install -y ngrok

# Verify ngrok installation
NGROK_PATH=$(which ngrok)
if [ -z "$NGROK_PATH" ]; then
    error "ngrok installation failed"
    exit 1
fi
log "ngrok installed at: $NGROK_PATH"

# Configure authtoken
if [ -n "${1:-}" ]; then
    AUTHTOKEN= "**********"
    log "Using provided authtoken"
else
    echo ""
    info "Get your authtoken from: "**********"://dashboard.ngrok.com/get-started/your-authtoken"
    info "After logging in, copy the authtoken from step 2"
    echo ""
    read -p "Enter your ngrok authtoken: "**********"
fi

# Add authtoken to ngrok config
log "Configuring ngrok authtoken..."
ngrok config add-authtoken "$AUTHTOKEN"

# Get current SSH port from our jumphost config
SSH_PORT=$(sudo grep -E "^Port" /etc/ssh/sshd_config.d/99-jumphost.conf 2>/dev/null | awk '{print $2}' || \
           sudo grep -E "^Port" /etc/ssh/sshd_config 2>/dev/null | awk '{print $2}' || \
           echo "22")
log "Detected SSH running on port: $SSH_PORT"

# Create systemd service for auto-start
log "Creating systemd service..."
sudo tee /etc/systemd/system/ngrok-jumphost.service > /dev/null << EOF
[Unit]
Description=Ngrok SSH Tunnel for Jump Host
After=network-online.target ssh.service
Wants=network-online.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER
ExecStart=$NGROK_PATH tcp $SSH_PORT --log=stdout
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ngrok-jumphost

# Security
NoNewPrivileges=true
PrivateTmp=true

# Environment
Environment="USER=$USER"
Environment="HOME=/home/$USER"

[Install]
WantedBy=multi-user.target
EOF

# Create monitoring script
log "Creating monitoring script..."
mkdir -p ~/bin

cat > ~/bin/ngrok-status << 'EOF'
#!/bin/bash
# Check ngrok tunnel status and get connection info

echo "=== Ngrok Jump Host Status ==="
echo ""

# Check if service is running
if systemctl is-active --quiet ngrok-jumphost; then
    echo -e "\033[0;32m✓ Ngrok service is running\033[0m"
else
    echo -e "\033[0;31m✗ Ngrok service is not running\033[0m"
    echo "  Start with: sudo systemctl start ngrok-jumphost"
    exit 1
fi

# Try to get tunnel info from API
echo ""
echo "Fetching tunnel information..."

# Wait a moment for API to be ready
sleep 2

# Get tunnel URLs using curl and jq
TUNNELS=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null)

if [ $? -eq 0 ] && [ -n "$TUNNELS" ]; then
    # Check if jq is available
    if command -v jq &> /dev/null; then
        echo ""
        echo "Active tunnels:"
        echo "$TUNNELS" | jq -r '.tunnels[] | "  URL: \(.public_url)\n  Local: \(.config.addr)\n"'
        
        # Extract TCP URL for SSH command
        TCP_URL=$(echo "$TUNNELS" | jq -r '.tunnels[] | select(.proto=="tcp") | .public_url' | head -1)
        
        if [ -n "$TCP_URL" ]; then
            # Remove tcp:// prefix and split host:port
            TCP_ADDR=${TCP_URL#tcp://}
            HOST=$(echo "$TCP_ADDR" | cut -d: -f1)
            PORT=$(echo "$TCP_ADDR" | cut -d: -f2)
            
            echo "SSH Connection Commands:"
            echo "----------------------"
            echo ""
            echo "Direct connection:"
            echo -e "\033[0;36m  ssh -p $PORT $USER@$HOST\033[0m"
            echo ""
            echo "As jump host to internal server:"
            echo -e "\033[0;36m  ssh -J $USER@$HOST:$PORT user@internal-server\033[0m"
            echo ""
            echo "Add to SSH config (~/.ssh/config):"
            echo -e "\033[0;36m"
            cat << CONFIG
  Host rpi-jump
      HostName $HOST
      Port $PORT
      User $USER
      ServerAliveInterval 60
      ServerAliveCountMax 3

  Host internal-via-rpi
      HostName 192.168.1.100
      User internaluser
      ProxyJump rpi-jump
CONFIG
            echo -e "\033[0m"
        fi
    else
        # Fallback without jq - try to parse with grep/sed
        echo "Active tunnel URL:"
        echo "$TUNNELS" | grep -o '"public_url":"[^"]*' | cut -d'"' -f4
    fi
else
    echo "  Unable to connect to ngrok API"
    echo "  The tunnel may still be starting..."
    echo ""
    echo "Check logs with:"
    echo "  sudo journalctl -u ngrok-jumphost -f"
fi

echo ""
echo "Service logs (last 10 lines):"
sudo journalctl -u ngrok-jumphost -n 10 --no-pager
EOF

chmod +x ~/bin/ngrok-status

# Make sure ~/bin is in PATH
if ! echo "$PATH" | grep -q "$HOME/bin"; then
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/bin:$PATH"
fi

# Create quick connection script
cat > ~/bin/ngrok-ssh << 'EOF'
#!/bin/bash
# Quick SSH command generator

# Get tunnel info
TUNNELS=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null)

if [ -n "$TUNNELS" ]; then
    if command -v jq &> /dev/null; then
        TCP_URL=$(echo "$TUNNELS" | jq -r '.tunnels[] | select(.proto=="tcp") | .public_url' | head -1)
    else
        TCP_URL=$(echo "$TUNNELS" | grep -o '"public_url":"tcp://[^"]*' | cut -d'"' -f4 | head -1)
    fi
    
    if [ -n "$TCP_URL" ]; then
        TCP_ADDR=${TCP_URL#tcp://}
        HOST=$(echo "$TCP_ADDR" | cut -d: -f1)
        PORT=$(echo "$TCP_ADDR" | cut -d: -f2)
        echo "ssh -p $PORT $USER@$HOST"
    else
        echo "No TCP tunnel found"
    fi
else
    echo "Ngrok not running or API not accessible"
fi
EOF

chmod +x ~/bin/ngrok-ssh

# Update firewall to allow ngrok API (local only)
log "Updating firewall rules..."
sudo ufw allow from 127.0.0.1 to any port 4040 comment 'Ngrok local API' > /dev/null 2>&1 || true

# Enable and start the service
log "Enabling ngrok service..."
sudo systemctl daemon-reload
sudo systemctl enable ngrok-jumphost
sudo systemctl start ngrok-jumphost

# Wait for service to start
log "Waiting for ngrok to establish tunnel..."
sleep 5

# Test if ngrok is working
if curl -s http://localhost:4040/api/tunnels > /dev/null 2>&1; then
    log "Ngrok API is responding"
else
    warning "Ngrok API not accessible yet, service may still be starting"
fi

# Show status
echo ""
log "Setup complete! Checking status..."
echo ""
~/bin/ngrok-status

# Create information file
cat > ~/NGROK_JUMPHOST_INFO.txt << EOF
Ngrok Jump Host Setup Complete!
===============================

Your SSH jump host is now accessible from anywhere via ngrok.

QUICK COMMANDS:
--------------
Get SSH command:       ngrok-ssh
Check status:          ngrok-status
View logs:            sudo journalctl -u ngrok-jumphost -f
Restart service:      sudo systemctl restart ngrok-jumphost
Stop service:         sudo systemctl stop ngrok-jumphost

IMPORTANT NOTES:
---------------
1. SSH ForceCommand logging has been disabled for ngrok compatibility
   - Audit logging is still active via auditd
   - All connections are still logged in /var/log/auth.log
   
2. The ngrok URL changes on each restart (free tier)
   - Run 'ngrok-status' after reboot to get new URL
   - Consider paid plan for static addresses

3. Security considerations:
   - Your SSH is now exposed to the internet
   - Fail2ban is still active (3 attempts = 1 hour ban)
   - Key-only authentication is still enforced
   - Monitor logs regularly for suspicious activity

TROUBLESHOOTING:
---------------
If ngrok doesn't start:
1. Check your authtoken: "**********"
2. View detailed logs: sudo journalctl -u ngrok-jumphost -n 50
3. Test manually: ngrok tcp 22
4. Check if port is in use: sudo netstat -tlnp | grep :22

To re-enable session logging:
1. sudo nano /etc/ssh/sshd_config.d/99-jumphost.conf
2. Uncomment the ForceCommand line
3. sudo systemctl reload ssh
(This will break ngrok connections)

EOF

echo ""
log "Installation complete!"
echo ""
info "Quick commands:"
info "  Get SSH command: ngrok-ssh"
info "  Check status: ngrok-status"
echo ""
warning "Your jump host is now accessible from the internet!"
warning "The ngrok URL will change on each restart (free tier)"estart (free tier)"