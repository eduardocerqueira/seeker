#date: 2024-12-24T16:41:37Z
#url: https://api.github.com/gists/2dab713a5a974c8907b2ee3d30d87cef
#owner: https://api.github.com/users/btc100k

#!/bin/bash

apt-get -y update && apt-get -y upgrade
apt-get -y install strongswan xl2tpd libstrongswan-standard-plugins libstrongswan-extra-plugins
service strongswan-starter stop
service xl2tpd stop
ipsec stop


VPN_SERVER_IP='<server ip>'
VPN_IPSEC_PSK='<server psk>'
VPN_USER='<vpn username>'
VPN_PASSWORD= "**********"
VPN_IP_SUBNET='<such as 192.168.1.0/24>'

cat > /etc/ipsec.conf <<EOF
config setup

conn %default
  rekeymargin=3m
  keyingtries=4
  keyexchange=ikev1
  authby=psk

conn VPN1
  left=%defaultroute
  right=$VPN_SERVER_IP
  ike=aes256-sha256-modp2048,aes128-sha256-modp2048,aes256-sha1-modp1024,aes128-sha1-modp1024
  esp=aes256-sha256-modp2048,aes128-sha256-modp2048,aes256-sha1-modp1024,aes128-sha1-modp1024
  auto=add
  dpddelay=30
  dpdtimeout=120
  dpdaction=clear
  rekey=yes
  ikelifetime=1h
  keylife=1h
  type=transport
  leftprotoport=17/1701
  rightid=$VPN_SERVER_IP
  rightprotoport=17/1701
EOF

cat > /etc/ipsec.secrets <<EOF
: PSK "$VPN_IPSEC_PSK"
EOF

chmod 600 /etc/ipsec.secrets

cat > /etc/xl2tpd/xl2tpd.conf <<EOF
[lac VPN1]
lns = $VPN_SERVER_IP
ppp debug = yes
pppoptfile = /etc/ppp/options.l2tpd.client
length bit = yes
EOF

cat > /etc/ppp/options.l2tpd.client <<EOF
# ipcp-accept-local
# ipcp-accept-remote
refuse-eap
# noccp
noauth
idle 1800
mtu 1410
mru 1410
# noipdefault
# defaultroute
replacedefaultroute
usepeerdns
debug
# require-mschap-v2
# connect-delay 5000
name $VPN_USER
password $VPN_PASSWORD
EOF

chmod 600 /etc/ppp/options.l2tpd.client

cat > /etc/ppp/chap-secrets <<EOF
# Secrets for authentication using CHAP
# client    server    secret          IP addresses
$VPN_USER VPN1 $VPN_PASSWORD *
# added via script (create_vpn.txt)
EOF

chmod 600 /etc/ppp/chap-secrets

cat > /etc/ppp/pap-secrets <<EOF
# Secrets for authentication using CHAP
# client    server    secret          IP addresses
$VPN_USER VPN1 $VPN_PASSWORD *
# added via script (create_vpn.txt)
EOF

chmod 600 /etc/ppp/pap-secrets


service strongswan-starter restart
service xl2tpd restart
ipsec restart

cat > /usr/local/bin/start-vpn <<EOF
#!/bin/bash

(service strongswan-starter start ;
sleep 2 ;
service xl2tpd start) && (

ipsec up VPN1
echo "c VPN1" > /var/run/xl2tpd/l2tp-control
sleep 5
#ip route add 10.0.0.0/24 dev ppp0
while ! ip link show ppp0 &>/dev/null; do
    echo "Waiting for ppp0 ..."
    sleep 1
done
ip route add $VPN_IP_SUBNET dev ppp0
)
EOF
chmod +x /usr/local/bin/start-vpn

cat > /usr/local/bin/stop-vpn <<EOF
#!/bin/bash

echo "Removing ppp0 route..."
ip route del $VPN_IP_SUBNET dev ppp0
(echo "d VPN1" > /var/run/xl2tpd/l2tp-control
ipsec down VPN1) && (
service xl2tpd stop ;
service strongswan-starter stop)
EOF
chmod +x /usr/local/bin/stop-vpn

echo "To start VPN type: start-vpn"
echo "To stop VPN type: stop-vpn"pn"