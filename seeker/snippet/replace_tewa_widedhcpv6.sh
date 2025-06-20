#date: 2025-06-20T17:00:33Z
#url: https://api.github.com/gists/ce9358684aec6413b06f8d0356ab5d4b
#owner: https://api.github.com/users/nexplorer-3e

#!/bin/sh
# TODO: not hardcoded host duid path etc
DHCP6C_PID="$(busybox pgrep dhcp6c)"
DHCP6C_PATH="$(ls /proc/$DHCP6C_PID/exe -l | awk '{print $11}')"

timeout="10"
my_dhcp6s_path="$PWD/dhcp6s"
my_dhcp6c_path="$PWD/dhcp6c"


while true; do
  sleep $timeout
  DHCP6S_PID="$(busybox pgrep dhcp6s)"
  DHCP6S_PATH="$(ls /proc/$DHCP6S_PID/exe -l | awk '{print $11}')"
  RADVD_PID="$(busybox pgrep radvd)"
  if [ -z "$RADVD_PID" ]; then
    continue
  fi
  # if radvd is started then arg should be okay
  if [ "$DHCP6S_PATH" = "$my_dhcp6s_path" ]; then
    continue
  fi
  # 2 cases: not active / not ours
  # deal with the latter
  if [ -n "$DHCP6S_PID" ]; then
    kill $DHCP6S_PID
  fi

  # then
  lifetime="$(awk '$1 == "AdvPreferredLifetime" { print $2 }' /var/radvd.conf)"  # with a semicolumn at end
  prefix="$(awk '$1 == "route" { printf "%-3s", $2 }' /var/radvd.conf | awk -F"/" '{ print $1 }')"
  prefixlen="$(ip -6 addr show dev br0 |awk '$2 ~ /2[0-9a-f]+/ {print $2}' | awk -F"/" '{print $2}')"
  pdargs="host kamisama { duid 00:01:ac:be:07:21; prefix $prefix/$prefixlen $lifetime };"
  cp /var/dhcp6s.conf d6spd.conf
  echo "$pdargs" >> d6spd.conf
  $my_dhcp6s_path -c d6spd.conf br0
done
