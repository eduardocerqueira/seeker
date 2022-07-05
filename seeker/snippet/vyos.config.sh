#date: 2022-07-05T16:45:56Z
#url: https://api.github.com/gists/4dfe24590a56434139744fb7d1bc6ce9
#owner: https://api.github.com/users/usrbinkat

set service dhcp-server shared-network-name LAN subnet 192.168.1.0/24 static-mapping talos-cp01 mac-address de:00:00:00:01:de
set service dhcp-server shared-network-name LAN subnet 192.168.1.0/24 static-mapping talos-cp01 ip-address 192.168.1.71
set system static-host-mapping host-name talos-cp01.home.arpa inet 192.168.1.71


set service dhcp-server shared-network-name LAN subnet 192.168.1.0/24 static-mapping talos-cp02 mac-address de:00:00:00:02:de
set service dhcp-server shared-network-name LAN subnet 192.168.1.0/24 static-mapping talos-cp02 ip-address 192.168.1.72
set system static-host-mapping host-name talos-cp02.home.arpa inet 192.168.1.72

set service dhcp-server shared-network-name LAN subnet 192.168.1.0/24 static-mapping talos-cp03 mac-address de:00:00:00:03:de
set service dhcp-server shared-network-name LAN subnet 192.168.1.0/24 static-mapping talos-cp03 ip-address 192.168.1.73
set system static-host-mapping host-name talos-cp03.home.arpa inet 192.168.1.73

set system static-host-mapping host-name talos-kubevirt.home.arpa inet 192.168.1.70