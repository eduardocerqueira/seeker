#date: 2024-12-04T17:13:34Z
#url: https://api.github.com/gists/027b894cf5d6eadcf9bc6406fd9a9dba
#owner: https://api.github.com/users/MorningLightMountain713

#!/bin/bash

umask 077

if [[ "$EUID" -ne 0 ]]; then
    echo "Please run as root / with sudo"
    exit
fi

! dpkg -s wireguard 2>&1 >/dev/null && apt get install wireguard

sed -i "s/#net.ipv4.ip_forward=1/net.ipv4.ip_forward=1/" /etc/sysctl.conf
sysctl -p >/dev/null

local_ip=10.11.12.1/31
peer_ip=10.11.12.0/31
vpn_port=51001

user=$(who am i | awk '{print $1}')
wan_int=$(ip route show default | awk '/default/ {print $5}')
wan_ip=$(curl -4 -s -L ipshow.me)

local_private=$(wg genkey)
local_public=$(echo $local_private | wg pubkey)
peer_private=$(wg genkey)
peer_public=$(echo $peer_private | wg pubkey)
psk=$(wg genpsk)

echo $peer_private >opnsense_private
echo $local_public >vps_public
echo $psk >opnsense_psk

chown "$user:$user" opnsense_private vps_public opnsense_psk

cat <<EOF >/etc/wireguard/wg_flux.conf
[Interface]
Address = $local_ip
PrivateKey = $local_private
ListenPort = $vpn_port
PostUp = /etc/wireguard/action_script up
PostDown = /etc/wireguard/action_script down

[Peer]
PublicKey = $peer_public
PreSharedKey = $psk
AllowedIPs = $peer_ip
EOF

cat <<EOF >/etc/wireguard/action_script
#!/bin/bash

state=\$1

[[ "\$state" = "up" ]] && action="-I" || action="-D"

iptables \$action INPUT -p udp -m udp --dport $vpn_port -j ACCEPT
iptables -t nat \$action PREROUTING -i $wan_int -p udp -m udp ! --dport $vpn_port -j DNAT --to-destination ${peer_ip%%/*}
iptables -t nat \$action PREROUTING -i $wan_int -p tcp -m tcp ! --dport 22 -j DNAT --to-destination ${peer_ip%%/*}
iptables -t nat \$action POSTROUTING -o $wan_int -j SNAT --to-source $wan_ip
EOF

chmod +x /etc/wireguard/action_script

systemctl enable wg-quick@wg_flux
systemctl daemon-reload

systemctl start wg-quick@wg_flux
