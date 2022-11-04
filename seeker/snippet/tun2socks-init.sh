#date: 2022-11-04T16:59:40Z
#url: https://api.github.com/gists/ab7269b052dd6053cc7b2f321b6a05c4
#owner: https://api.github.com/users/aMir733

#!/usr/bin/env bash

trap "cleanup" INT

DEFAULT_VPN=37.32.0.0/16
GATEWAY="$(ip route get 1.1.1.1 | sed -n 's/.* via \([^\ ]*\) .*/\1/p')"
[[ ${#GATEWAY} == 0 ]] && exit 1

cleanup() {
    ip link set dev tun0 down
    systemctl restart NetworkManager
    exit 0
}

if ! ip link set dev tun0 up &>/dev/null ; then
    ip tuntap add mode tun dev tun0
    ip addr add 198.18.0.1/15 dev tun0
    ip link set dev tun0 up
fi

ip route del default
ip route add ${1:-$DEFAULT_VPN} via $GATEWAY dev wlan0 metric 1
ip route add default via 198.18.0.1 dev tun0 metric 2

echo 'nameserver 1.1.1.1' > /etc/resolv.conf

run_tun() {
    kill -9 $PID_TUN 2>/dev/null
    tun2socks -device tun0 -proxy socks5://127.0.0.1:1080 &
    PID_TUN=$(pgrep tun2socks)
}
run_tun
while true ; do # Fight the memory leak(???)
    [[ $(ps -p $PID_TUN -o %cpu= | grep --color=none -Po '^\s*\K[0-9]+') -gt 7 ]] && run_tun
    [[ $(ps -p $(pgrep v2ray) -o %cpu= | grep --color=none -Po '^\s*\K[0-9]+') -gt 25 ]] && systemctl restart v2ray@client
    sleep 15
done