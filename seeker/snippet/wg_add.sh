#date: 2023-02-01T16:50:53Z
#url: https://api.github.com/gists/5645c70416070e2b55bb647b22047fa0
#owner: https://api.github.com/users/aeifn

#!/usr/bin/env bash

N=$1
NAME=$2
SERVER_ADDR=vpn.mathem.ru

SERVER_PRIVATE_KEY=$(</etc/wireguard/private.key) 
SERVER_PUBLIC_KEY=$(wg pubkey</etc/wireguard/private.key)

CLIENT_PRIVATE_KEY=$(wg genkey)
CLIENT_PUBLIC_KEY=$(echo $CLIENT_PRIVATE_KEY | wg pubkey)

cat >> /etc/wireguard/wg0.conf << EOF
[Peer] 
# Name = $NAME
PublicKey = $CLIENT_PUBLIC_KEY
AllowedIPs = 10.8.0.$N/32
PersistentKeepalive = 25
EOF

CONF="$PWD/wg0.$N.$NAME.conf"

cat > "$CONF" << EOF
[Interface]
PrivateKey = $CLIENT_PRIVATE_KEY
Address = 10.8.0.$N/24
DNS = 10.129.0.2

[Peer]
PublicKey = $SERVER_PUBLIC_KEY
AllowedIPs = 10.0.0.0/8
Endpoint = $SERVER_ADDR:51820
EOF

echo Saved to "$CONF"
wg syncconf wg0 <(wg-quick strip wg0)
qrencode -t ansiutf8 < "$CONF"
