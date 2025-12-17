#date: 2025-12-17T17:20:00Z
#url: https://api.github.com/gists/064bd1e3e2643f1f5b3031ae4be9abc1
#owner: https://api.github.com/users/biturbo1337

#!/bin/sh
set -eu

INSTALL_DIR="/etc/zapret2"
IPSET_DIR="$INSTALL_DIR/ipset"
HOSTLIST="$INSTALL_DIR/mylist-domru.txt"
CONFIG_FILE="$INSTALL_DIR/config"
INIT_FILE="/etc/init.d/zapret2"
UPDATE_SCRIPT="/usr/bin/update_zapret2_ipsets"

QUEUE_NUM="200"
ENABLE_IPV6=1
IPSET_FLUSH=1

[ "$(id -u)" -eq 0 ] || { echo "ERROR: run as root"; exit 1; }

opkg update >/dev/null
opkg install curl jq ipset nftables kmod-nfnetlink-queue kmod-nft-queue >/dev/null

mkdir -p "$INSTALL_DIR" "$IPSET_DIR"

# ===== Download zapret2 (GitHub API) WITHOUT jq regex/test() =====
REL_JSON="$(curl -fsSL https://api.github.com/repos/bol-van/zapret2/releases/latest)"

ASSET_URL="$(echo "$REL_JSON" | jq -r '.assets[].browser_download_url' \
  | grep -m1 'openwrt-embedded\.tar\.gz$' || true)"

if [ -z "$ASSET_URL" ]; then
  ASSET_URL="$(echo "$REL_JSON" | jq -r '.assets[].browser_download_url' \
    | grep -m1 '\.tar\.gz$' || true)"
fi

[ -n "$ASSET_URL" ] || { echo "ERROR: zapret2 release asset not found"; exit 1; }

TMP="$(mktemp -d)"
curl -fL "$ASSET_URL" -o "$TMP/zapret2.tar.gz"
tar -xzf "$TMP/zapret2.tar.gz" -C "$TMP"

NFQWS2_BIN="$(find "$TMP" -type f -name nfqws2 2>/dev/null | head -n 1 || true)"
[ -n "$NFQWS2_BIN" ] || { echo "ERROR: nfqws2 not found in archive"; rm -rf "$TMP"; exit 1; }

# Copy full package (lua/blobs included)
cp -a "$TMP"/* "$INSTALL_DIR/" 2>/dev/null || true
cp -f "$NFQWS2_BIN" "$INSTALL_DIR/nfqws2"
chmod +x "$INSTALL_DIR/nfqws2"
rm -rf "$TMP"

ZAPRET_LIB="$(find "$INSTALL_DIR" -maxdepth 4 -type f -name zapret-lib.lua | head -n 1 || true)"
ZAPRET_ANTI="$(find "$INSTALL_DIR" -maxdepth 4 -type f -name zapret-antidpi.lua | head -n 1 || true)"
[ -n "$ZAPRET_LIB" ] && [ -n "$ZAPRET_ANTI" ] || { echo "ERROR: lua files not found"; exit 1; }

# ===== Hostlist (YouTube + Discord) =====
cat > "$HOSTLIST" <<'EOF'
# YouTube
youtube.com
youtu.be
googlevideo.com
ytimg.com
yt3.ggpht.com
ggpht.com
youtubei.googleapis.com
youtube.googleapis.com
youtube-ui.l.google.com
youtube-nocookie.com
wide-youtube.l.google.com

# Discord
discord.com
discord.gg
discordapp.com
discordcdn.com
discordapp.net
discord.media
gateway.discord.gg
cdn.discordapp.com
media.discordapp.net
EOF

# ===== IP sets =====
touch "$IPSET_DIR/cloudflare_ips.set" "$IPSET_DIR/aws_ips.set"
if [ "$ENABLE_IPV6" -eq 1 ]; then
  touch "$IPSET_DIR/cloudflare_ips_v6.set" "$IPSET_DIR/aws_ips_v6.set"
fi

create_ipsets() {
  ipset create cloudflare_ips hash:net family inet  -exist
  ipset create aws_ips        hash:net family inet  -exist
  if [ "$ENABLE_IPV6" -eq 1 ]; then
    ipset create cloudflare_ips_v6 hash:net family inet6 -exist
    ipset create aws_ips_v6        hash:net family inet6 -exist
  fi
}

update_ipsets() {
  create_ipsets

  if [ "$IPSET_FLUSH" -eq 1 ]; then
    ipset flush cloudflare_ips || true
    ipset flush aws_ips || true
    if [ "$ENABLE_IPV6" -eq 1 ]; then
      ipset flush cloudflare_ips_v6 || true
      ipset flush aws_ips_v6 || true
    fi
  fi

  curl -fsSL https://www.cloudflare.com/ips-v4 | while read -r net; do
    [ -n "$net" ] && ipset add cloudflare_ips "$net" -exist
  done

  curl -fsSL https://ip-ranges.amazonaws.com/ip-ranges.json | jq -r '.prefixes[].ip_prefix' | while read -r net; do
    [ -n "$net" ] && ipset add aws_ips "$net" -exist
  done

  if [ "$ENABLE_IPV6" -eq 1 ]; then
    curl -fsSL https://www.cloudflare.com/ips-v6 | while read -r net; do
      [ -n "$net" ] && ipset add cloudflare_ips_v6 "$net" -exist
    done

    curl -fsSL https://ip-ranges.amazonaws.com/ip-ranges.json | jq -r '.ipv6_prefixes[].ipv6_prefix' | while read -r net; do
      [ -n "$net" ] && ipset add aws_ips_v6 "$net" -exist
    done
  fi

  ipset save cloudflare_ips > "$IPSET_DIR/cloudflare_ips.set"
  ipset save aws_ips        > "$IPSET_DIR/aws_ips.set"
  if [ "$ENABLE_IPV6" -eq 1 ]; then
    ipset save cloudflare_ips_v6 > "$IPSET_DIR/cloudflare_ips_v6.set"
    ipset save aws_ips_v6        > "$IPSET_DIR/aws_ips_v6.set"
  fi
}

# ===== Standalone ipset updater (ash-safe) =====
cat > "$UPDATE_SCRIPT" <<EOF
#!/bin/sh
set -eu

ENABLE_IPV6=$ENABLE_IPV6
IPSET_FLUSH=$IPSET_FLUSH
IPSET_DIR="$IPSET_DIR"

ipset create cloudflare_ips hash:net family inet  -exist
ipset create aws_ips        hash:net family inet  -exist
if [ "\$ENABLE_IPV6" -eq 1 ]; then
  ipset create cloudflare_ips_v6 hash:net family inet6 -exist
  ipset create aws_ips_v6        hash:net family inet6 -exist
fi

if [ "\$IPSET_FLUSH" -eq 1 ]; then
  ipset flush cloudflare_ips || true
  ipset flush aws_ips || true
  if [ "\$ENABLE_IPV6" -eq 1 ]; then
    ipset flush cloudflare_ips_v6 || true
    ipset flush aws_ips_v6 || true
  fi
fi

curl -fsSL https://www.cloudflare.com/ips-v4 | while read -r net; do [ -n "\$net" ] && ipset add cloudflare_ips "\$net" -exist; done
curl -fsSL https://ip-ranges.amazonaws.com/ip-ranges.json | jq -r '.prefixes[].ip_prefix' | while read -r net; do [ -n "\$net" ] && ipset add aws_ips "\$net" -exist; done

if [ "\$ENABLE_IPV6" -eq 1 ]; then
  curl -fsSL https://www.cloudflare.com/ips-v6 | while read -r net; do [ -n "\$net" ] && ipset add cloudflare_ips_v6 "\$net" -exist; done
  curl -fsSL https://ip-ranges.amazonaws.com/ip-ranges.json | jq -r '.ipv6_prefixes[].ipv6_prefix' | while read -r net; do [ -n "\$net" ] && ipset add aws_ips_v6 "\$net" -exist; done
fi

mkdir -p "\$IPSET_DIR"
ipset save cloudflare_ips > "\$IPSET_DIR/cloudflare_ips.set"
ipset save aws_ips        > "\$IPSET_DIR/aws_ips.set"
if [ "\$ENABLE_IPV6" -eq 1 ]; then
  ipset save cloudflare_ips_v6 > "\$IPSET_DIR/cloudflare_ips_v6.set"
  ipset save aws_ips_v6        > "\$IPSET_DIR/aws_ips_v6.set"
fi

echo OK
EOF
chmod +x "$UPDATE_SCRIPT"

# ===== nfqws2 config =====
cat > "$CONFIG_FILE" <<EOF
QUEUE_NUM="$QUEUE_NUM"

NFQWS2_ARGS="--qnum \$QUEUE_NUM --debug \\
  --lua-init=@$ZAPRET_LIB \\
  --lua-init=@$ZAPRET_ANTI \\
  --filter-tcp=443 --filter-l7=tls --hostlist=$HOSTLIST \\
    --payload=tls_client_hello \\
      --lua-desync=fake:blob=fake_default_tls:tcp_md5:repeats=11:tls_mod=rnd,rndsni,dupsid,padencap:ip_autottl=-1,3-20:ip6_autottl=-1,3-20 \\
    --payload=tls_client_hello \\
      --lua-desync=multisplit:pos=1:seqovl=5:seqovl_pattern=0x1603030000 \\
    --new \\
  --filter-udp=443 --filter-l7=quic --hostlist=$HOSTLIST \\
    --payload=quic_initial \\
      --lua-desync=fake:blob=fake_default_quic:repeats=11:ip_autottl=-1,3-20:ip6_autottl=-1,3-20 \\
    --new"
EOF

# ===== nftables + NFQUEUE =====
sysctl -w net.netfilter.nf_conntrack_tcp_be_liberal=1 >/dev/null || true

nft delete table inet zapret2 2>/dev/null || true
nft add table inet zapret2
nft add chain inet zapret2 post "{ type filter hook postrouting priority 101; }"
nft add chain inet zapret2 pre  "{ type filter hook prerouting priority -101; }"

# outbound: CF/AWS (1-12)
nft add rule inet zapret2 post meta mark and 0x40000000 == 0 ip  daddr @cloudflare_ips tcp dport 443 ct original packets 1-12 queue num "$QUEUE_NUM" bypass
nft add rule inet zapret2 post meta mark and 0x40000000 == 0 ip  daddr @aws_ips        tcp dport 443 ct original packets 1-12 queue num "$QUEUE_NUM" bypass
nft add rule inet zapret2 post meta mark and 0x40000000 == 0 ip  daddr @cloudflare_ips udp dport 443 ct original packets 1-12 queue num "$QUEUE_NUM" bypass
nft add rule inet zapret2 post meta mark and 0x40000000 == 0 ip  daddr @aws_ips        udp dport 443 ct original packets 1-12 queue num "$QUEUE_NUM" bypass

# outbound: generic narrow intercept (1-4)
nft add rule inet zapret2 post meta mark and 0x40000000 == 0 tcp dport 443 ct original packets 1-4 queue num "$QUEUE_NUM" bypass
nft add rule inet zapret2 post meta mark and 0x40000000 == 0 udp dport 443 ct original packets 1-4 queue num "$QUEUE_NUM" bypass

# outbound IPv6
if [ "$ENABLE_IPV6" -eq 1 ]; then
  nft add rule inet zapret2 post meta mark and 0x40000000 == 0 ip6 daddr @cloudflare_ips_v6 tcp dport 443 ct original packets 1-12 queue num "$QUEUE_NUM" bypass
  nft add rule inet zapret2 post meta mark and 0x40000000 == 0 ip6 daddr @aws_ips_v6        tcp dport 443 ct original packets 1-12 queue num "$QUEUE_NUM" bypass
  nft add rule inet zapret2 post meta mark and 0x40000000 == 0 ip6 daddr @cloudflare_ips_v6 udp dport 443 ct original packets 1-12 queue num "$QUEUE_NUM" bypass
  nft add rule inet zapret2 post meta mark and 0x40000000 == 0 ip6 daddr @aws_ips_v6        udp dport 443 ct original packets 1-12 queue num "$QUEUE_NUM" bypass

  nft add rule inet zapret2 post meta mark and 0x40000000 == 0 ip6 nexthdr tcp tcp dport 443 ct original packets 1-4 queue num "$QUEUE_NUM" bypass
  nft add rule inet zapret2 post meta mark and 0x40000000 == 0 ip6 nexthdr udp udp dport 443 ct original packets 1-4 queue num "$QUEUE_NUM" bypass
fi

# reply
nft add rule inet zapret2 pre meta mark and 0x40000000 == 0 tcp sport "{80,443}" ct reply packets 1-12 queue num "$QUEUE_NUM" bypass
nft add rule inet zapret2 pre meta mark and 0x40000000 == 0 udp sport "{443}"    ct reply packets 1-12 queue num "$QUEUE_NUM" bypass

# ===== procd service =====
cat > "$INIT_FILE" <<'EOF'
#!/bin/sh /etc/rc.common
USE_PROCD=1
START=99
STOP=10

start_service() {
  for f in /etc/zapret2/ipset/*.set; do
    [ -f "$f" ] && ipset restore -exist < "$f" || true
  done

  . /etc/zapret2/config

  procd_open_instance
  procd_set_param command /etc/zapret2/nfqws2 $NFQWS2_ARGS
  procd_set_param respawn 3600 5 5
  procd_set_param stdout 1
  procd_set_param stderr 1
  procd_set_param stderr 1
  procd_close_instance
}

stop_service() {
  killall nfqws2 2>/dev/null || true
}
EOF
chmod +x "$INIT_FILE"

update_ipsets
/etc/init.d/zapret2 enable
/etc/init.d/zapret2 restart || /etc/init.d/zapret2 start

echo "OK: installed"
echo "config:   $CONFIG_FILE"
echo "hostlist: $HOSTLIST"
echo "update:   $UPDATE_SCRIPT"
echo "logs:     logread -e zapret2"
