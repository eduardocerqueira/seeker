#date: 2025-07-04T17:06:19Z
#url: https://api.github.com/gists/912761ce1197950fdfcc8503b5ac16fb
#owner: https://api.github.com/users/MichaelSaucier

#!/bin/sh

set -e

# Configuration
LAN_IF="igb1"
TAILSCALE_SUBNET="100.64.0.0/10"
ANCHOR_FILE="/usr/local/etc/pfanchors/tailscale_anchor.conf"
RC_SERVICE="/usr/local/etc/rc.d/tailscale-nat"

# --- Functions ---

install() {
  echo "🔧 Enabling IP forwarding..."
  sysctl net.inet.ip.forwarding=1
  grep -q 'net.inet.ip.forwarding' /etc/sysctl.conf || echo 'net.inet.ip.forwarding=1' >> /etc/sysctl.conf

  echo "📁 Creating Tailscale NAT anchor rule for $LAN_IF..."
  mkdir -p /usr/local/etc/pfanchors
  cat <<EOF > "$ANCHOR_FILE"
nat on $LAN_IF from $TAILSCALE_SUBNET to any -> ($LAN_IF)
EOF

  echo "🧠 Creating pf anchor loader service..."
  cat <<'EOF' > "$RC_SERVICE"
#!/bin/sh
#
# PROVIDE: tailscale_nat
# REQUIRE: NETWORKING
# BEFORE:  LOGIN

. /etc/rc.subr

name="tailscale_nat"
rcvar="tailscale_nat_enable"
start_cmd="${name}_start"

tailscale_nat_start()
{
    echo "Loading Tailscale pf anchor..."
    pfctl -a tailscale -f /usr/local/etc/pfanchors/tailscale_anchor.conf
}

load_rc_config $name
run_rc_command "$1"
EOF

  chmod +x "$RC_SERVICE"

  echo "✅ Enabling tailscale-nat to auto-run at boot..."
  grep -q 'tailscale_nat_enable' /etc/rc.conf || echo 'tailscale_nat_enable="YES"' >> /etc/rc.conf

  echo "📦 Loading anchor now..."
  pfctl -a tailscale -f "$ANCHOR_FILE"
  pfctl -e || true

  echo "✅ Install complete. Subnet routing from Tailscale ($TAILSCALE_SUBNET) to LAN ($LAN_IF) is now active and persistent."
}

uninstall() {
  echo "🔄 Flushing pf anchor rules for tailscale..."
  pfctl -a tailscale -Fs nat || true

  echo "🧽 Removing pf anchor rule file..."
  rm -f "$ANCHOR_FILE"

  echo "🧽 Removing rc.d service script..."
  rm -f "$RC_SERVICE"

  echo "🧼 Cleaning up /etc/rc.conf..."
  sed -i '' '/tailscale_nat_enable/d' /etc/rc.conf || true

  echo "🧼 Cleaning up /etc/sysctl.conf..."
  sed -i '' '/net.inet.ip.forwarding/d' /etc/sysctl.conf || true

  echo "🛑 Disabling IP forwarding immediately..."
  sysctl net.inet.ip.forwarding=0

  echo "🧹 Uninstall complete. Tailscale NAT and forwarding removed from system."
}

# --- Main ---

case "$1" in
  --uninstall)
    uninstall
    ;;
  --install|"")
    install
    ;;
  *)
    echo "Usage: $0 [--install | --uninstall]"
    exit 1
    ;;
esac