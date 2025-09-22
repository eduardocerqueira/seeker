#date: 2025-09-22T17:00:41Z
#url: https://api.github.com/gists/4754921a8e720f2674c5e563b56c73a8
#owner: https://api.github.com/users/txase

#/bin/bash -e -x

LOOKUP_DOMAIN=google.com

podman kill firefox-captive >/dev/null 2>&1

echo "Checking if we can resolve ${LOOKUP_DOMAIN}..."
if getent hosts "$LOOKUP_DOMAIN" >/dev/null 2>&1; then
  echo "Success! No need to launch captive portal."
  sleep 3
  exit 0
fi

# --- Wi-Fi Interface autodetect if not set ---
if [[ -z "$IFACE" ]]; then
  IFACE=$(iw dev 2>/dev/null | awk '$1=="Interface"{print $2}' | head -n1)
  if [[ -z "$IFACE" ]]; then
    echo "Error: Could not auto-detect Wi-Fi interface. Use --iface to specify."
    exit 1
  fi
fi

# --- Get DNS server from NetworkManager for the Wi-Fi interface ---
DNS=$(
  nmcli device show "$IFACE" 2>/dev/null | awk '/IP.\.DNS/ {print $2; exit}'
)
if [[ -z "$DNS" ]]; then
  echo "Error: Could not get DNS server for interface '$IFACE'. Are you connected via Wi-Fi?"
  exit 1
fi

echo "Using DNS server: $DNS"

podman run --rm -d --name firefox-captive \
  --dns "$DNS" \
  -e FF_OPEN_URL=http://captive.apple.com/hotspot-detect.html \
  -e FF_KIOSK=1 \
  -p 38370:5800 \
  --pull never \
  docker.io/jlesage/firefox

echo "Opening http://localhost:38370..."
xdg-open http://localhost:38370 >/dev/null 2>&1

echo "Waiting for DNS resolution to succeed for ${LOOKUP_DOMAIN}..."
until getent hosts "$LOOKUP_DOMAIN" >/dev/null 2>&1; do
  sleep 1
done

echo "Success!"
podman kill firefox-captive

sleep 3