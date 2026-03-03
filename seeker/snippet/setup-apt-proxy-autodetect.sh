#date: 2026-03-03T17:26:33Z
#url: https://api.github.com/gists/aaabb412542ed27d458dc79baeba9381
#owner: https://api.github.com/users/thostetler

#!/usr/bin/env bash
set -euo pipefail

# Creates:
#  - /etc/apt/apt.conf.d/02proxy
#  - /usr/bin/apt-proxy-detect.sh
#
# Based on the two files in the referenced gist:
#  - 02proxy sets Proxy-Auto-Detect to /usr/bin/apt-proxy-detect.sh and Acquire::Retries 0
#  - apt-proxy-detect.sh checks reachability of apt-cacher via nc and prints proxy URL or DIRECT
#
# Usage:
#   sudo ./setup-apt-proxy-autodetect.sh <APT_CACHER_IP> [PORT]
#
# Example:
#   sudo ./setup-apt-proxy-autodetect.sh 192.168.88.1
#   sudo ./setup-apt-proxy-autodetect.sh 192.168.88.1 3142

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "ERROR: must be run as root (use sudo)." >&2
  exit 1
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <APT_CACHER_IP> [PORT]" >&2
  exit 1
fi

APT_CACHER_IP="$1"
APT_CACHER_PORT="${2:-3142}"

# Basic IPv4 validation
if ! [[ "$APT_CACHER_IP" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
  echo "ERROR: '$APT_CACHER_IP' does not look like an IPv4 address." >&2
  exit 1
fi
IFS='.' read -r o1 o2 o3 o4 <<<"$APT_CACHER_IP"
for o in "$o1" "$o2" "$o3" "$o4"; do
  if (( o < 0 || o > 255 )); then
    echo "ERROR: '$APT_CACHER_IP' is not a valid IPv4 address." >&2
    exit 1
  fi
done

# Port validation
if ! [[ "$APT_CACHER_PORT" =~ ^[0-9]+$ ]] || (( APT_CACHER_PORT < 1 || APT_CACHER_PORT > 65535 )); then
  echo "ERROR: '$APT_CACHER_PORT' is not a valid TCP port (1-65535)." >&2
  exit 1
fi

CONF_PATH="/etc/apt/apt.conf.d/02proxy"
SCRIPT_PATH="/usr/bin/apt-proxy-detect.sh"

backup_if_exists() {
  local p="$1"
  if [[ -e "$p" ]]; then
    local ts
    ts="$(date +%Y%m%d-%H%M%S)"
    cp -a "$p" "${p}.bak.${ts}"
    echo "Backed up $p -> ${p}.bak.${ts}"
  fi
}

backup_if_exists "$CONF_PATH"
backup_if_exists "$SCRIPT_PATH"

# Write /etc/apt/apt.conf.d/02proxy (matches gist structure/content)
cat >"$CONF_PATH" <<'EOF'
## /etc/apt/apt.conf.d/02proxy

# You can add this line for faster failover
Acquire::Retries 0;
# Make sure you use the full path
Acquire::http::Proxy-Auto-Detect "/usr/bin/apt-proxy-detect.sh";
EOF

chmod 0644 "$CONF_PATH"
echo "Wrote $CONF_PATH"

# Write /usr/bin/apt-proxy-detect.sh, injecting IP and port
cat >"$SCRIPT_PATH" <<EOF
#!/bin/bash
## Tells Proxy-Auto-Config if a specified proxy is reachable or if
## it should just fallback to a direct connection.
##
## This file should be executable and referenced by its full path
## in the /etc/apt/apt.conf.d/02proxy file we create too.

ip="${APT_CACHER_IP}"
port=${APT_CACHER_PORT}
## This will install netcat automatically if it's missing if uncommented
#if [[ \$(which nc >/dev/null; echo \$?) -ne 0 ]]; then
# apt install -y netcat-traditional
#fi

if [[ \$(nc -w1 -z \$ip \$port &>/dev/null; echo \$?) -eq 0 ]]; then
 echo -n "http://\${ip}:\${port}/"
else
 echo -n "DIRECT"
fi
EOF

chmod 0755 "$SCRIPT_PATH"
echo "Wrote $SCRIPT_PATH (executable)"

echo
echo "Done."
echo "Note: apt-proxy-detect.sh uses 'nc' (netcat). Install it if missing, e.g.:"
echo "  sudo apt-get update && sudo apt-get install -y netcat-traditional"
echo
echo "Optional quick check (will print proxy URL or DIRECT):"
echo "  $SCRIPT_PATH"