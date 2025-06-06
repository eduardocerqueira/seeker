#date: 2025-06-06T17:02:46Z
#url: https://api.github.com/gists/409d5c4fafdfe9bc04a9165593de9316
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

# Install ngrok using apt repository
log "Installing ngrok via apt..."
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
    | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
    && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
    | sudo tee /etc/apt/sources.list.d/ngrok.list \
    && sudo apt update \
    && sudo apt install -y ngrok

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
SSH_PORT=$(sudo grep -E "^Port" /etc/ssh/sshd_config.d/99-jumphost.conf 2>/dev/null | awk '{print $2}' || echo "22")
log "Detected SSH running on port: $SSH_PORT"

# Create ngrok configuration file
log "Creating ngrok configuration..."
mkdir -p ~/.config/ngrok

cat > ~/.config/ngrok/ngrok.yml << EOF
version: "2"
authtoken: "**********"

tunnels:
  ssh-jumphost:
    proto: tcp
    addr: $SSH_PORT
    metadata:
      name: "RPI5 Secure Jump Host"
    
  ssh-direct:
    proto: tcp
    addr: $SSH_PORT
    # This creates a second tunnel as backup
    
web_addr: 127.0.0.1:4040
EOF

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
ExecStart=/usr/local/bin/ngrok start ssh-jumphost --config /home/$USER/.config/ngrok/ngrok.yml --log stdout
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

# Get tunnel URLs
TUNNELS=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null)

if [ $? -eq 0 ] && [ -n "$TUNNELS" ]; then
    echo ""
    echo "Active tunnels:"
    echo "$TUNNELS" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for tunnel in data.get('tunnels', []):
        print(f\"  Name: {tunnel['name']}\")
        print(f\"  URL: {tunnel['public_url']}\")
        print(f\"  Local: {tunnel['config']['addr']}\")
        print()
except:
    print('  Unable to parse tunnel data')
"
    
    # Extract TCP URL for SSH command
    TCP_URL=$(echo "$TUNNELS" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for tunnel in data.get('tunnels', []):
        if tunnel['proto'] == 'tcp':
            url = tunnel['public_url'].replace('tcp://', '')
            print(url)
            break
except:
    pass
" 2>/dev/null)
    
    if [ -n "$TCP_URL" ]; then
        HOST=$(echo "$TCP_URL" | cut -d: -f1)
        PORT=$(echo "$TCP_URL" | cut -d: -f2)
        echo "SSH Connection Commands:"
        echo "----------------------"
        echo ""
        echo "Direct connection:"
        echo -e "\033[0;36m  ssh -p $PORT $USER@$HOST\033[0m"
        echo ""
        echo "As jump host to internal server:"
        echo -e "\033[0;36m  ssh -J $USER@$HOST:$PORT user@internal-server\033[0m"
        echo ""
        echo "In SSH config (~/.ssh/config):"
        echo -e "\033[0;36m"
        echo "  Host rpi-jump"
        echo "      HostName $HOST"
        echo "      Port $PORT"
        echo "      User $USER"
        echo ""
        echo "  Host internal-via-rpi"
        echo "      HostName 192.168.1.100"
        echo "      User internaluser"
        echo "      ProxyJump rpi-jump"
        echo -e "\033[0m"
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
mkdir -p ~/bin
if ! grep -q "$HOME/bin" ~/.bashrc; then
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
fi

# Create quick connection script
log "Creating connection helper..."
cat > ~/bin/ngrok-connect-info << 'EOF'
#!/bin/bash
# Get connection info in a format easy to copy/paste

TCP_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for tunnel in data.get('tunnels', []):
        if tunnel['proto'] == 'tcp':
            url = tunnel['public_url'].replace('tcp://', '')
            print(url)
            break
except:
    pass
" 2>/dev/null)

if [ -n "$TCP_URL" ]; then
    HOST=$(echo "$TCP_URL" | cut -d: -f1)
    PORT=$(echo "$TCP_URL" | cut -d: -f2)
    echo "ssh -p $PORT $USER@$HOST"
else
    echo "Tunnel not active. Run: ngrok-status"
fi
EOF

chmod +x ~/bin/ngrok-connect-info

# Update firewall to allow ngrok API (local only)
log "Updating firewall rules..."
sudo ufw allow from 127.0.0.1 to any port 4040 comment 'Ngrok local API'

# Enable and start the service
log "Enabling ngrok service..."
sudo systemctl daemon-reload
sudo systemctl enable ngrok-jumphost
sudo systemctl start ngrok-jumphost

# Wait for service to start
log "Waiting for ngrok to establish tunnel..."
sleep 5

# Show status
echo ""
log "Setup complete! Checking status..."
echo ""
source ~/.bashrc
ngrok-status

# Create information file
cat > ~/NGROK_JUMPHOST_INFO.txt << EOF
Ngrok Jump Host Setup Complete!
===============================

Your SSH jump host is now accessible from anywhere via ngrok.

COMMANDS:
---------
Check status and get connection info:
  ngrok-status

Get just the SSH command:
  ngrok-connect-info

View logs:
  sudo journalctl -u ngrok-jumphost -f

Restart service:
  sudo systemctl restart ngrok-jumphost

Stop service:
  sudo systemctl stop ngrok-jumphost

SECURITY NOTES:
--------------
1. Ngrok exposes your SSH port to the internet with a random URL
2. Your existing jump host security (fail2ban, keys-only, etc) still applies
3. The ngrok URL changes on each restart unless you have a paid plan
4. Monitor logs for any suspicious activity

USAGE EXAMPLES:
--------------
1. Direct SSH:
   ssh -p [port] $USER@[ngrok-host]

2. Jump to internal server:
   ssh -J $USER@[ngrok-host]:[port] user@internal-server

3. SCP through jump host:
   scp -J $USER@[ngrok-host]:[port] file.txt user@internal-server:/tmp/

4. Port forwarding:
   ssh -J $USER@[ngrok-host]:[port] -L 8080:internal-server:80 user@internal-server

PAID FEATURES:
-------------
Consider ngrok paid plans for:
- Static TCP addresses (URL doesn't change)
- Custom domains
- IP restrictions
- Higher connection limits

EOF

log "Installation complete!"
echo ""
info "Run 'ngrok-status' to see your connection details"
info "Your ngrok tunnel will automatically start on boot"
echo ""
warning "Remember: Your jump host is now accessible from the internet!"
warning "Monitor logs regularly: sudo journalctl -u ngrok-jumphost -f"umphost -f"