#date: 2025-10-24T17:08:29Z
#url: https://api.github.com/gists/effae0a0f9ce92f3f57344b01fd9a8d9
#owner: https://api.github.com/users/anhphuongtranle99z-del

#!/usr/bin/env bash
set -euo pipefail

# Script: install-dante.sh
# Mục đích: Cài Dante SOCKS5 với user/pass (tranleaku / AnhPhuong123@)
# Chạy trên Ubuntu/Debian (root)

PROXY_USER="tranleaku"
PROXY_PASS="AnhPhuong123@"
SOCKS_PORT=1080
CONF_FILE="/etc/danted.conf"
SERVICE_NAME="danted"

# 1) Kiểm tra root
if [ "$(id -u)" -ne 0 ]; then
  echo "Run as root (sudo). Exiting."
  exit 1
fi

echo ">>> Update apt and install dante-server..."
apt-get update -y
DEBIAN_FRONTEND=noninteractive apt-get install -y dante-server || {
  echo "Failed to install dante-server via apt. Exiting."
  exit 1
}

# 2) Detect default network interface (tries a couple methods)
detect_iface() {
  # Try ip route method
  IFACE=$(ip route get 8.8.8.8 2>/dev/null | awk -F 'dev ' '{print $2}' | awk '{print $1}' || true)
  if [ -n "$IFACE" ]; then
    echo "$IFACE"
    return 0
  fi
  # Fallback to parsing default route table
  IFACE=$(ip -o -4 route show to default | awk '{print $5; exit}' || true)
  if [ -n "$IFACE" ]; then
    echo "$IFACE"
    return 0
  fi
  # Last resort, pick first non-loopback UP interface
  IFACE=$(ip -o link show | awk -F': ' '{print $2}' | egrep -v 'lo|virbr|docker|veth' | head -n1)
  echo "${IFACE:-eth0}"
}

IFACE="$(detect_iface)"
echo ">>> Detected network interface: $IFACE"

# 3) Create proxy user if not exists, set password
if id -u "$PROXY_USER" >/dev/null 2>&1; then
  echo "User $PROXY_USER already exists. Updating password..."
else
  echo "Creating user $PROXY_USER (no shell, no home)..."
  useradd -M -s /usr/sbin/nologin "$PROXY_USER"
fi

echo "Setting password for $PROXY_USER..."
echo "${PROXY_USER}:${PROXY_PASS}" | chpasswd

# 4) Write danted.conf
cat > "$CONF_FILE" <<EOF
logoutput: syslog
internal: ${IFACE} port = ${SOCKS_PORT}
external: ${IFACE}

method: username
user.notprivileged: nobody

client pass {
    from: 0.0.0.0/0 to: 0.0.0.0/0
    log: connect disconnect error
}

pass {
    from: 0.0.0.0/0 to: 0.0.0.0/0
    protocol: tcp udp
    log: connect disconnect error
}
EOF

chmod 640 "$CONF_FILE"
chown root:root "$CONF_FILE"

# 5) Restart + enable service
echo ">>> Restarting and enabling $SERVICE_NAME..."
systemctl daemon-reload
systemctl restart "$SERVICE_NAME" || {
  echo "Service failed to start. Showing journal lines:"
  journalctl -u "$SERVICE_NAME" -n 50 --no-pager
  exit 1
}
systemctl enable "$SERVICE_NAME"

# 6) Open firewall port: try ufw first, else add iptables rule
if command -v ufw >/dev/null 2>&1; then
  if ufw status | grep -q inactive; then
    echo "ufw installed but inactive. Skipping ufw allow."
  else
    echo "Adding ufw allow ${SOCKS_PORT}/tcp"
    ufw allow "${SOCKS_PORT}/tcp"
  fi
else
  echo "ufw not found, adding iptables rule to ACCEPT ${SOCKS_PORT}/tcp temporarily."
  iptables -I INPUT -p tcp --dport "${SOCKS_PORT}" -j ACCEPT
  # Note: this iptables rule is NOT persistent across reboots unless netfilter-persistent is used
  if command -v netfilter-persistent >/dev/null 2>&1; then
    netfilter-persistent save
  fi
fi

# 7) Verify listening
sleep 1
if ss -tulnp | grep -q ":${SOCKS_PORT}"; then
  echo ">>> Dante appears to be listening on port ${SOCKS_PORT}."
else
  echo ">>> Warning: Dante does NOT appear to be listening on port ${SOCKS_PORT}. Check status:"
  systemctl status "$SERVICE_NAME" --no-pager
  exit 1
fi

# 8) Print summary & test suggestion
IP_ADDR="$(ip -4 addr show "$IFACE" | grep -oP '(?<=inet\s)\d+(\.\d+){3}')"
echo "================================================================"
echo "Dante SOCKS5 installed and running."
echo "Proxy endpoint: ${IP_ADDR}:${SOCKS_PORT}"
echo "Username: ${PROXY_USER}"
echo "Password: "**********"
echo
echo "On Windows/Proxifier: "**********":${SOCKS_PORT}, enable Authentication and enter username/password."
echo
echo "To view service status: systemctl status ${SERVICE_NAME}"
echo "To view logs: journalctl -u ${SERVICE_NAME} -f"
echo "================================================================"
============"
