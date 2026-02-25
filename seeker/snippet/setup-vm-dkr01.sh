#date: 2026-02-25T17:42:58Z
#url: https://api.github.com/gists/76c69972d00f9d0cee3a7d3a15d4d8d1
#owner: https://api.github.com/users/k1ll4p1x3l

#!/usr/bin/env bash
set -euo pipefail

# VM: vm-dkr01.lab.stbhuber.com (Debian 13.x)
# Purpose: minimal Docker worker + Tailscale, DHCP networking, Break-Glass root only.
# Notes:
# - DHCP only (static mapping handled on OPNsense)
# - Docker rootful for now (rootless + IPA user later)
# - Fixes APT first (disables cdrom sources, writes official Debian repos in deb822 format)

FQDN="vm-dkr01.lab.stbhuber.com"
HOST_SHORT="vm-dkr01"
DOMAIN="lab.stbhuber.com"

die()  { echo "FATAL: $*" >&2; exit 1; }
info() { echo "INFO:  $*" >&2; }

require_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    die "Bitte als root ausführen."
  fi
}

require_root

info "0) APT reparieren: cdrom deaktivieren + offizielle Debian-Repos setzen (deb822)"

# a) 'cdrom:' Quellen deaktivieren (kann in sources.list und/oder *.list vorkommen)
if [[ -f /etc/apt/sources.list ]]; then
  sed -i 's/^[[:space:]]*deb[[:space:]]\+cdrom:/# deb cdrom:/g' /etc/apt/sources.list || true
  sed -i 's/^[[:space:]]*deb-src[[:space:]]\+cdrom:/# deb-src cdrom:/g' /etc/apt/sources.list || true
fi

if compgen -G "/etc/apt/sources.list.d/*.list" > /dev/null; then
  for f in /etc/apt/sources.list.d/*.list; do
    sed -i 's/^[[:space:]]*deb[[:space:]]\+cdrom:/# deb cdrom:/g' "$f" || true
    sed -i 's/^[[:space:]]*deb-src[[:space:]]\+cdrom:/# deb-src cdrom:/g' "$f" || true
  done
fi

# b) Debian Codename ermitteln (Debian 13 = trixie; wir nehmen, was das System meldet)
. /etc/os-release
CODENAME="${VERSION_CODENAME:-}"
[[ -n "$CODENAME" ]] || die "VERSION_CODENAME nicht gefunden in /etc/os-release"

install -m 0755 -d /etc/apt/sources.list.d

# Offizielle Quellen im deb822-Format (debian.sources)
cat > /etc/apt/sources.list.d/debian.sources <<EOF
Types: deb
URIs: https://deb.debian.org/debian
Suites: ${CODENAME} ${CODENAME}-updates
Components: main non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg

Types: deb
URIs: https://security.debian.org/debian-security
Suites: ${CODENAME}-security
Components: main non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
EOF

# Option: alte sources.list stilllegen, um doppelte Quellen zu vermeiden (nicht löschen)
if [[ -f /etc/apt/sources.list ]]; then
  sed -i 's/^[[:space:]]*deb[[:space:]]/## disabled: deb /g' /etc/apt/sources.list || true
  sed -i 's/^[[:space:]]*deb-src[[:space:]]/## disabled: deb-src /g' /etc/apt/sources.list || true
fi

apt-get update

info "1) Basispakete installieren (minimal + headless tools)"
apt-get install -y --no-install-recommends \
  ca-certificates curl gnupg nano \
  openssh-server \
  qemu-guest-agent \
  iproute2 iputils-ping dnsutils

systemctl enable --now qemu-guest-agent >/dev/null 2>&1 || true

info "2) Hostname/FQDN setzen + /etc/hosts pflegen"
hostnamectl set-hostname "$FQDN"

# Debian-typisch: 127.0.1.1 für den eigenen Hostnamen
if ! grep -qE "^\s*127\.0\.1\.1\s+$FQDN\s+$HOST_SHORT(\s|$)" /etc/hosts; then
  sed -i '/^\s*127\.0\.1\.1\s\+/d' /etc/hosts
  echo "127.0.1.1 $FQDN $HOST_SHORT" >> /etc/hosts
fi

info "3) Netzwerk auf DHCP konfigurieren (OPNsense macht das statische Mapping)"

# Ziel: DHCP robust erzwingen, egal ob networkd oder ifupdown aktiv ist.
# Wir bevorzugen systemd-networkd, wenn verfügbar (deterministischer als Mischbetrieb).
if systemctl list-unit-files | grep -q '^systemd-networkd\.service'; then
  info "   systemd-networkd verfügbar -> aktiviere DHCP via networkd"
  apt-get install -y --no-install-recommends systemd-resolved

  systemctl enable --now systemd-networkd systemd-resolved

  mkdir -p /etc/systemd/network
  cat > /etc/systemd/network/10-dhcp.network <<'EOF'
[Match]
Name=en* eth*

[Network]
DHCP=yes

[DHCPv4]
UseDNS=true
UseDomains=true
EOF

  ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf
  systemctl restart systemd-networkd systemd-resolved
else
  info "   systemd-networkd nicht vorhanden -> nutze ifupdown DHCP"
  apt-get install -y --no-install-recommends ifupdown

  # Default-Interface ermitteln
  DEV="$(ip -4 route show default 0.0.0.0/0 2>/dev/null | awk '{print $5}' | head -n1 || true)"
  if [[ -z "$DEV" ]]; then
    DEV="$(ls /sys/class/net | grep -E '^(en|eth)' | head -n1 || true)"
  fi
  [[ -n "$DEV" ]] || die "Konnte Netzwerk-Interface nicht erkennen. Bitte manuell prüfen."

  cat > /etc/network/interfaces <<EOF
# Managed by setup script for $FQDN (DHCP)
auto lo
iface lo inet loopback

allow-hotplug $DEV
iface $DEV inet dhcp
EOF

  systemctl restart networking || true
fi

info "4) Tailscale installieren (offizielles Repo)"
install -m 0755 -d /usr/share/keyrings

curl -fsSL "https://pkgs.tailscale.com/stable/debian/${CODENAME}.noarmor.gpg" \
  -o /usr/share/keyrings/tailscale-archive-keyring.gpg
curl -fsSL "https://pkgs.tailscale.com/stable/debian/${CODENAME}.tailscale-keyring.list" \
  -o /etc/apt/sources.list.d/tailscale.list

apt-get update
apt-get install -y tailscale
systemctl enable --now tailscaled

info "   Hinweis: 'tailscale up' machst du bewusst selbst (Auth/ACL/Tags)."

info "5) Docker Engine installieren (offizielles Docker-Repo, rootless später)"
apt-get remove -y docker.io docker-compose docker-doc podman-docker containerd runc 2>/dev/null || true

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

ARCH="$(dpkg --print-architecture)"
cat > /etc/apt/sources.list.d/docker.list <<EOF
deb [arch=${ARCH} signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian ${CODENAME} stable
EOF

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

systemctl enable --now docker

info "6) Abschlusschecks"
info "   - Hostname: $(hostname -f || true)"
info "   - IPs (DHCP):"
ip -4 addr show || true
info "   - Default Route:"
ip -4 route show default || true
info "   - Tailscale service: $(systemctl is-active tailscaled || true)"
info "   - Docker service: $(systemctl is-active docker || true)"
info "   - Docker version:"
docker version 2>/dev/null || true

info "FERTIG. Nächste manuelle Schritte:"
echo "  1) OPNsense: DHCP static mapping prüfen/setzen (vm-dkr01 -> 10.10.20.11)."
echo "  2) Tailscale anmelden: sudo tailscale up (mit deinen üblichen Flags/ACL/Tags)."
echo "  3) Docker testen: docker run --rm hello-world"
echo "  4) Danach: FreeIPA Join + später Rootless Docker (wenn IPA-User existiert)."
