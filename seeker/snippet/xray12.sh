#date: 2025-05-16T17:00:54Z
#url: https://api.github.com/gists/fa9aa0982e097572e0ab9f09daa0ea30
#owner: https://api.github.com/users/leowph

#!/bin/bash

# Xray VLESS REALITY Installer Script
# Installs Xray-core and configures VLESS over TCP with REALITY.
# Supports dynamic IP update information for EC2-like environments.
# Includes enhanced script execution logging.

# --- Configuration ---
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file for the script's execution
SCRIPT_LOG_FILE="/var/log/xray_installer.log"

# --- Helper Functions ---
# Function to log messages to both console and script log file
_log_message() {
    local type="$1"
    local color="$2"
    local message="$3"
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")

    # Console output
    echo -e "${color}[${type}] ${NC}${message}"
    
    # File output (without color codes)
    echo "${timestamp} [${type}] ${message}" >> "${SCRIPT_LOG_FILE}"
}

log_info() {
    _log_message "INFO" "${BLUE}" "$1"
}

log_success() {
    _log_message "SUCCESS" "${GREEN}" "$1"
}

log_warning() {
    _log_message "WARNING" "${YELLOW}" "$1"
}

log_error() {
    # Error messages go to stderr for console
    >&2 echo -e "${RED}[ERROR] ${NC}$1"
    # Also log to file
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "${timestamp} [ERROR] $1" >> "${SCRIPT_LOG_FILE}"
}


# --- Sanity Checks ---
# Exit on error
set -e
# Exit on pipefail
set -o pipefail

# Initialize log file
# Must be done before any log_info/error calls if they depend on SCRIPT_LOG_FILE being writable
# Ensure log directory exists and set permissions if needed.
# For /var/log, root privileges are generally sufficient.
touch "${SCRIPT_LOG_FILE}" # Create if it doesn't exist
chmod 644 "${SCRIPT_LOG_FILE}" # Set permissions

echo -e "${BLUE}[INIT] ${NC}This script's execution log will be saved to: ${YELLOW}${SCRIPT_LOG_FILE}${NC}"
echo "$(date "+%Y-%m-%d %H:%M:%S") [INIT] Script execution started. Log file: ${SCRIPT_LOG_FILE}" >> "${SCRIPT_LOG_FILE}"


# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    log_error "This script must be run as root. Please use sudo."
    exit 1
fi

# --- Main Installation Logic ---
main() {
    log_info "Starting Xray VLESS REALITY installation..."

    # 1. System Update and Dependencies
    log_info "Updating system packages and installing dependencies (apt update, apt upgrade, curl, unzip, jq, coreutils)..."
    apt update > >(tee -a "${SCRIPT_LOG_FILE}") 2> >(tee -a "${SCRIPT_LOG_FILE}" >&2)
    log_info "apt update completed."
    apt upgrade -y > >(tee -a "${SCRIPT_LOG_FILE}") 2> >(tee -a "${SCRIPT_LOG_FILE}" >&2)
    log_info "apt upgrade completed."
    apt install -y curl unzip jq coreutils > >(tee -a "${SCRIPT_LOG_FILE}") 2> >(tee -a "${SCRIPT_LOG_FILE}" >&2)
    log_success "System updated and dependencies installed."

    # 2. Get Initial Server IP (for display purposes, might change)
    log_info "Attempting to fetch server public IP..."
    CURRENT_SERVER_IP=$(curl -s https://ifconfig.me)
    if [ -z "$CURRENT_SERVER_IP" ]; then
        log_warning "Failed to fetch IP from ifconfig.me, trying api.ipify.org..."
        CURRENT_SERVER_IP=$(curl -s https://api.ipify.org)
    fi

    if [ -z "$CURRENT_SERVER_IP" ]; then
        log_warning "Could not automatically determine server IP. You may need to find it manually."
        CURRENT_SERVER_IP="YOUR_SERVER_IP" # Placeholder
    else
        log_info "Current server public IP detected as: $CURRENT_SERVER_IP"
    fi

    # 3. User Inputs
    log_info "Requesting user inputs for Xray configuration."
    
    DEFAULT_XRAY_PORT="443"
    read -p "$(echo -e "${YELLOW}Enter Xray listening port (default: $DEFAULT_XRAY_PORT): ${NC}")" XRAY_PORT
    XRAY_PORT=${XRAY_PORT:-$DEFAULT_XRAY_PORT}
    log_info "User input for XRAY_PORT: $XRAY_PORT"
    # Basic port validation
    if ! [[ "$XRAY_PORT" =~ ^[0-9]+$ ]] || [ "$XRAY_PORT" -lt 1 ] || [ "$XRAY_PORT" -gt 65535 ]; then
        log_error "Invalid port number: '$XRAY_PORT'. Please enter a number between 1 and 65535."
        exit 1
    fi

    DEFAULT_REALITY_SNI="www.microsoft.com"
    read -p "$(echo -e "${YELLOW}Enter REALITY destination server name (SNI, e.g., $DEFAULT_REALITY_SNI): ${NC}")" REALITY_SNI
    REALITY_SNI=${REALITY_SNI:-$DEFAULT_REALITY_SNI}
    log_info "User input for REALITY_SNI: $REALITY_SNI"
    if [ -z "$REALITY_SNI" ]; then
        log_error "REALITY SNI cannot be empty."
        exit 1
    fi
    # Simple check for domain-like pattern (not foolproof)
    if ! [[ "$REALITY_SNI" =~ ^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
        log_warning "The entered SNI '$REALITY_SNI' might not be a valid domain name. Please ensure it's correct."
    fi
    log_info "Using SNI: $REALITY_SNI (Ensure this domain resolves and serves HTTPS on port 443)"


    # 4. Download and Install Xray-core
    log_info "Preparing to download and install Xray-core..."
    XRAY_INSTALL_DIR="/usr/local"
    XRAY_BIN_DIR="$XRAY_INSTALL_DIR/bin"
    XRAY_SHARE_DIR="$XRAY_INSTALL_DIR/share/xray"
    XRAY_ETC_DIR="$XRAY_INSTALL_DIR/etc/xray"
    # XRAY_LOG_DIR="/var/log/xray" # Xray's own logs, not script logs

    log_info "Creating Xray directories: $XRAY_BIN_DIR, $XRAY_SHARE_DIR, $XRAY_ETC_DIR"
    mkdir -p "$XRAY_BIN_DIR" "$XRAY_SHARE_DIR" "$XRAY_ETC_DIR" # Removed $XRAY_LOG_DIR as it's for Xray app logs

    log_info "Fetching latest Xray-core release URL for linux-64 from GitHub API..."
    LATEST_XRAY_URL=$(curl -s "https://api.github.com/repos/XTLS/Xray-core/releases/latest" | jq -r '.assets[] | select(.name == "Xray-linux-64.zip") | .browser_download_url' | head -n 1)

    if [ -z "$LATEST_XRAY_URL" ] || [ "$LATEST_XRAY_URL" == "null" ]; then
        log_error "Failed to get the latest Xray-core release URL for Xray-linux-64.zip. Check network or GitHub API status, or if the asset name has changed."
        exit 1
    fi
    log_info "Latest Xray-core URL: $LATEST_XRAY_URL"

    log_info "Downloading Xray-core from $LATEST_XRAY_URL to xray_core.zip..."
    curl -L -o xray_core.zip "$LATEST_XRAY_URL"
    if [ $? -ne 0 ]; then
        log_error "Failed to download Xray-core. Curl exit code: $?."
        exit 1
    fi
    log_success "Xray-core downloaded successfully."

    TMP_XRAY_DIR="/tmp/xray_install_temp_$(date +%s)" # Unique temp dir
    log_info "Creating temporary directory for unzipping: $TMP_XRAY_DIR"
    mkdir -p "$TMP_XRAY_DIR"
    
    log_info "Unzipping xray_core.zip to $TMP_XRAY_DIR..."
    unzip -o xray_core.zip -d "$TMP_XRAY_DIR" > >(tee -a "${SCRIPT_LOG_FILE}") 2> >(tee -a "${SCRIPT_LOG_FILE}" >&2)
    log_success "Xray-core unzipped."
    
    log_info "Installing Xray binaries and data files..."
    install -m 755 "$TMP_XRAY_DIR/xray" "$XRAY_BIN_DIR/xray"
    install -m 644 "$TMP_XRAY_DIR/geoip.dat" "$XRAY_SHARE_DIR/geoip.dat"
    install -m 644 "$TMP_XRAY_DIR/geosite.dat" "$XRAY_SHARE_DIR/geosite.dat"
    log_success "Xray files installed to $XRAY_BIN_DIR and $XRAY_SHARE_DIR."
    
    log_info "Cleaning up temporary files: $TMP_XRAY_DIR and xray_core.zip..."
    rm -rf "$TMP_XRAY_DIR" xray_core.zip
    log_success "Temporary files cleaned up."
    log_success "Xray-core installed successfully to $XRAY_BIN_DIR/xray"

    # 5. Generate Credentials
    log_info "Generating Xray credentials (UUID, X25519 key pair, Short ID)..."
    XRAY_UUID=$("$XRAY_BIN_DIR/xray" uuid)
    
    KEY_PAIR_OUTPUT=$("$XRAY_BIN_DIR/xray" x25519)
    PRIVATE_KEY=$(echo "$KEY_PAIR_OUTPUT" | grep "Private key:" | awk '{print $3}')
    PUBLIC_KEY=$(echo "$KEY_PAIR_OUTPUT" | grep "Public key:" | awk '{print $3}')

    SHORT_ID=$(head /dev/urandom | tr -dc 'a-f0-9' | head -c 16)

    if [ -z "$XRAY_UUID" ] || [ -z "$PRIVATE_KEY" ] || [ -z "$PUBLIC_KEY" ] || [ -z "$SHORT_ID" ]; then
        log_error "Failed to generate all necessary credentials. UUID: $XRAY_UUID, PrivateKey: $PRIVATE_KEY, PublicKey: $PUBLIC_KEY, ShortID: $SHORT_ID"
        exit 1
    fi
    log_success "Credentials generated."
    log_info "UUID: $XRAY_UUID"
    log_info "Private Key (SERVER ONLY): $PRIVATE_KEY"
    log_info "Public Key (for client): $PUBLIC_KEY"
    log_info "Short ID (for client): $SHORT_ID"

    # 6. Create Xray Configuration File
    log_info "Creating Xray configuration file at $XRAY_ETC_DIR/config.json..."
    mkdir -p "$XRAY_ETC_DIR" # Ensure directory exists, though already created
    
    cat > "$XRAY_ETC_DIR/config.json" << EOF
{
  "log": {
    "loglevel": "warning" 
  },
  "routing": {
    "domainStrategy": "AsIs",
    "rules": [
      {
        "type": "field",
        "outboundTag": "direct",
        "protocol": ["bittorrent"]
      },
      {
        "type": "field",
        "ip": ["geoip:private"],
        "outboundTag": "direct"
      }
    ]
  },
  "inbounds": [
    {
      "listen": "0.0.0.0",
      "port": ${XRAY_PORT},
      "protocol": "vless",
      "settings": {
        "clients": [
          {
            "id": "${XRAY_UUID}",
            "flow": "xtls-rprx-vision"
          }
        ],
        "decryption": "none"
      },
      "streamSettings": {
        "network": "tcp",
        "security": "reality",
        "realitySettings": {
          "show": false,
          "dest": "${REALITY_SNI}:443", 
          "xver": 0,
          "serverNames": ["${REALITY_SNI}"],
          "privateKey": "${PRIVATE_KEY}",
          "minClientVer": "", 
          "maxClientVer": "", 
          "maxTimeDiff": 60000,
          "shortIds": ["${SHORT_ID}"] 
        }
      },
      "sniffing": {
        "enabled": true,
        "destOverride": ["http", "tls", "fakedns"]
      }
    }
  ],
  "outbounds": [
    {
      "protocol": "freedom",
      "tag": "direct"
    },
    {
      "protocol": "blackhole",
      "tag": "block"
    }
  ]
}
EOF
    log_success "Xray configuration file created at $XRAY_ETC_DIR/config.json."

    # 7. Create Systemd Service File
    log_info "Creating systemd service file at /etc/systemd/system/xray.service..."
    IP_INFO_FILE="/var/run/xray_public_ip.txt" 

    cat > /etc/systemd/system/xray.service << EOF
[Unit]
Description=Xray Service
Documentation=https://github.com/XTLS/Xray-core
After=network.target nss-lookup.target

[Service]
User=nobody
CapabilityBoundingSet=CAP_NET_ADMIN CAP_NET_BIND_SERVICE
AmbientCapabilities=CAP_NET_ADMIN CAP_NET_BIND_SERVICE
NoNewPrivileges=true
ExecStartPre=/bin/bash -c 'mkdir -p /var/run/xray && (curl -s https://ifconfig.me > ${IP_INFO_FILE}.tmp || curl -s https://api.ipify.org > ${IP_INFO_FILE}.tmp || echo "IP_FETCH_FAILED" > ${IP_INFO_FILE}.tmp) && mv ${IP_INFO_FILE}.tmp ${IP_INFO_FILE} && chmod 644 ${IP_INFO_FILE}'
ExecStart=${XRAY_BIN_DIR}/xray run -config ${XRAY_ETC_DIR}/config.json
Restart=on-failure
RestartPreventExitStatus=23
LimitNPROC=10000
LimitNOFILE=1000000

[Install]
WantedBy=multi-user.target
EOF
    log_success "Systemd service file created at /etc/systemd/system/xray.service."

    # 8. Enable and Start Xray Service
    log_info "Reloading systemd daemon..."
    systemctl daemon-reload
    log_info "Enabling Xray service to start on boot..."
    systemctl enable xray.service
    log_info "Starting Xray service..."
    systemctl start xray.service

    log_info "Waiting for 5 seconds for Xray service to initialize..."
    sleep 5 
    if systemctl is-active --quiet xray.service; then
        log_success "Xray service started successfully and is active."
    else
        log_error "Xray service failed to start or is not active. Please check Xray logs: 'journalctl -u xray -e --no-pager' and the script log: '${SCRIPT_LOG_FILE}'. Also verify the Xray config: 'cat $XRAY_ETC_DIR/config.json'"
        # Optionally, dump last few lines of xray log to script log
        journalctl -u xray -n 20 --no-pager >> "${SCRIPT_LOG_FILE}"
        exit 1
    fi
    
    # 9. Display Client Configuration
    log_info "--- Client Configuration ---"
    
    SERVER_IP_FOR_CLIENT=$(cat "${IP_INFO_FILE}" 2>/dev/null)
    if [ "$SERVER_IP_FOR_CLIENT" == "IP_FETCH_FAILED" ] || [ -z "$SERVER_IP_FOR_CLIENT" ]; then
        log_warning "Could not read IP from ${IP_INFO_FILE}. Using initially detected IP ($CURRENT_SERVER_IP) or placeholder for client config display."
        SERVER_IP_FOR_CLIENT="$CURRENT_SERVER_IP" 
    else
        log_info "Server IP for client configuration (from ${IP_INFO_FILE}): ${SERVER_IP_FOR_CLIENT}"
    fi

    # Display to console only for this part
    echo -e "\n${YELLOW}Your Xray VLESS REALITY server is configured.${NC}"
    echo -e "Use the following details for your client:"
    echo -e "-----------------------------------------------------"
    echo -e "  ${GREEN}Address (Server IP):${NC}  ${SERVER_IP_FOR_CLIENT}"
    echo -e "  ${GREEN}Port:${NC}                 ${XRAY_PORT}"
    echo -e "  ${GREEN}UUID (ID):${NC}            ${XRAY_UUID}"
    echo -e "  ${GREEN}Protocol:${NC}             vless"
    echo -e "  ${GREEN}Flow:${NC}                 xtls-rprx-vision"
    echo -e "  ${GREEN}Security:${NC}             reality"
    echo -e "  ${GREEN}Network (Type):${NC}       tcp"
    echo -e "  ${GREEN}SNI (ServerName):${NC}     ${REALITY_SNI}"
    echo -e "  ${GREEN}Public Key (pbk):${NC}     ${PUBLIC_KEY}"
    echo -e "  ${GREEN}Short ID (sid):${NC}       ${SHORT_ID}"
    echo -e "  ${GREEN}SpiderX (fingerprint):${NC} chrome (or your preferred browser, e.g. firefox, safari, edge, random)"
    echo -e "-----------------------------------------------------"

    VLESS_URI="vless://${XRAY_UUID}@${SERVER_IP_FOR_CLIENT}:${XRAY_PORT}?type=tcp&security=reality&flow=xtls-rprx-vision&sni=${REALITY_SNI}&pbk=${PUBLIC_KEY}&sid=${SHORT_ID}#Xray-REALITY-${REALITY_SNI//./_}"

    echo -e "\n${YELLOW}VLESS Configuration URI (copy and paste into your client):${NC}"
    echo -e "${GREEN}${VLESS_URI}${NC}"
    echo -e "-----------------------------------------------------"

    # Log client config to script log as well
    log_info "Client Configuration Details:"
    log_info "  Address (Server IP):  ${SERVER_IP_FOR_CLIENT}"
    log_info "  Port:                 ${XRAY_PORT}"
    log_info "  UUID (ID):            ${XRAY_UUID}"
    log_info "  Protocol:             vless"
    log_info "  Flow:                 xtls-rprx-vision"
    log_info "  Security:             reality"
    log_info "  Network (Type):       tcp"
    log_info "  SNI (ServerName):     ${REALITY_SNI}"
    log_info "  Public Key (pbk):     ${PUBLIC_KEY}"
    log_info "  Short ID (sid):       ${SHORT_ID}"
    log_info "  SpiderX (fingerprint): chrome (or preferred)"
    log_info "  VLESS URI:            ${VLESS_URI}"

    # 10. Post-installation Notes (display to console, also log)
    log_info "--- Important Notes ---"
    NOTES_CONTENT="1. If your server's public IP address changes (e.g., after an EC2 instance reboot), \n"
    NOTES_CONTENT+="   you can find the current IP by checking the content of the file: ${IP_INFO_FILE} on the server.\n"
    NOTES_CONTENT+="   Command: cat ${IP_INFO_FILE}\n"
    NOTES_CONTENT+="   You will need to update the 'Address' in your client configuration manually.\n"
    NOTES_CONTENT+="2. To check Xray service status: systemctl status xray --no-pager\n"
    NOTES_CONTENT+="3. To view Xray service logs (not script logs): journalctl -u xray -e --no-pager\n"
    NOTES_CONTENT+="4. Xray configuration file: ${XRAY_ETC_DIR}/config.json\n"
    NOTES_CONTENT+="5. Make sure your firewall (e.g., ufw) allows incoming connections on TCP port ${XRAY_PORT}.\n"
    NOTES_CONTENT+="   Example for ufw: sudo ufw allow ${XRAY_PORT}/tcp\n"
    NOTES_CONTENT+="6. This script's execution log is saved at: ${SCRIPT_LOG_FILE}"

    echo -e "${NOTES_CONTENT//\\n/\n}" # Display notes with newlines
    # Log notes to file (replace color codes if any, though none here)
    echo -e "${NOTES_CONTENT//\\n/\n}" >> "${SCRIPT_LOG_FILE}"


    log_success "Installation and configuration complete!"
    log_info "Script finished successfully."
}

# Run the main function
main

exit 0
