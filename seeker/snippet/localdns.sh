#date: 2025-11-12T16:39:19Z
#url: https://api.github.com/gists/51dd70d78402be4f5e2e16afd2d9adcc
#owner: https://api.github.com/users/derekjc

#!/usr/bin/env bash
set -euo pipefail

CLOUD=$(sudo cloud-init query cloud-name)
echo "Detected cloud: ${CLOUD}"

echo "=== Installing Unbound ==="
sudo dnf install -y unbound

echo "=== Disabling other DNS services ==="
sudo systemctl disable --now systemd-resolved || true
sudo systemctl disable --now dnsmasq || true

sudo mkdir -p /etc/NetworkManager/conf.d

sudo tee /etc/NetworkManager/conf.d/90-dns-none.conf >/dev/null <<'EOF'
[main]
dns=none
EOF
sudo systemctl restart NetworkManager

CONF="/etc/unbound/conf.d/local.conf"
sudo tee "$CONF" >/dev/null <<EOF
server:
  interface: 0.0.0.0
  access-control: 127.0.0.0/8 allow
  cache-min-ttl: 60
  cache-max-ttl: 86400
  prefetch: yes
  msg-cache-size: 64m
  rrset-cache-size: 128m
  num-threads: 2
  verbosity: 1
  so-reuseport: yes
  do-ip6: no

# --- Default public resolvers ---
forward-zone:
  name: "."
  forward-addr: 1.1.1.1
  forward-addr: 8.8.8.8
  forward-addr: 9.9.9.9
EOF

case "$CLOUD" in
  azure)
    cat <<'EOF' | sudo tee -a "$CONF" >/dev/null

# --- Azure internal zones ---
forward-zone:
  name: "internal.cloudapp.net."
  forward-addr: 168.63.129.16
  forward-addr: 1.1.1.1
  forward-addr: 8.8.8.8

forward-zone:
  name: "azure.com."
  forward-addr: 168.63.129.16
  forward-addr: 1.1.1.1
  forward-addr: 8.8.8.8

forward-zone:
  name: "windows.net."
  forward-addr: 168.63.129.16
  forward-addr: 1.1.1.1
  forward-addr: 8.8.8.8
EOF
    ;;
  aws)
    cat <<'EOF' | sudo tee -a "$CONF" >/dev/null

# --- AWS internal zones ---
forward-zone:
  name: "compute.internal."
  forward-addr: 169.254.169.253
  forward-addr: 169.254.169.250
  forward-addr: 1.1.1.1
  forward-addr: 8.8.8.8
EOF
    ;;
  gce)
    cat <<'EOF' | sudo tee -a "$CONF" >/dev/null

# --- GCP internal zones ---
forward-zone:
  name: "c.internal."
  forward-addr: 169.254.169.254
  forward-addr: 1.1.1.1
  forward-addr: 8.8.8.8

forward-zone:
  name: "google.internal."
  forward-addr: 169.254.169.254
  forward-addr: 1.1.1.1
  forward-addr: 8.8.8.8
EOF
    ;;
  alicloud)
    cat <<'EOF' | sudo tee -a "$CONF" >/dev/null

# --- Alibaba Cloud internal zones ---
forward-zone:
  name: "aliyun.internal."
  forward-addr: 100.100.2.136
  forward-addr: 1.1.1.1
  forward-addr: 8.8.8.8
EOF
    ;;
  *)
    echo "# Unknown cloud provider, using only public resolvers" | sudo tee -a "$CONF" >/dev/null
    ;;
esac

echo "=== Validating Unbound configuration ==="
sleep 2
sudo unbound-checkconf || { echo "ERROR: Invalid Unbound configuration"; exit 1; }


echo "=== Enabling and starting Unbound ==="
sudo systemctl enable --now unbound

echo "=== Test DNS resolution ==="
dig +short google.com @127.0.0.1 || true

echo "=== Pointing system to localhost resolver ==="
sudo bash -c 'echo "nameserver 127.0.0.1" > /etc/resolv.conf'

echo "=== Done ==="
echo "Unbound installed with cloud-aware DNS forwarding for ${CLOUD}"
