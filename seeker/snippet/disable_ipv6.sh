#date: 2025-03-24T16:49:00Z
#url: https://api.github.com/gists/78ce6f906163f38b98644dcaaf663f7e
#owner: https://api.github.com/users/lulkien

#!/bin/ash


uci set 'network.lan.ipv6=0'
# uci set 'network.wan.ipv6=0'
uci set 'network.lan.delegate=0'
uci set 'dhcp.lan.dhcpv6=disabled'

uci -q delete dhcp.lan.dhcpv6
uci -q delete dhcp.lan.ra
uci -q delete network.globals.ula_prefix

uci commit dhcp
uci commit network

/etc/init.d/network restart