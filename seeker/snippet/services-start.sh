#date: 2022-05-18T17:17:08Z
#url: https://api.github.com/gists/6fbacfbcf401ca23db6baf4309fad1d5
#owner: https://api.github.com/users/chrisjian

#!/bin/sh

# multi SSID with VLAN script, for ASUS AC66U_B1 with merlin.
# DHCP service is configured by main router, such as ER-X or other devices,
# Not in this router.
#
# setup before hand:
#       set "router"  to "AP Mode"
#       this will be put all ports and wireless in br0
#       create a guest network ssid, exam: asus_guest_1
#       enable Administration => System => Enable JFFS custom scripts and configs
#    

# some basic info of original AP mode:

# exec 'brctl show' command in shell, then you will get info like below:
#
#        bridge name    bridge id               STP enabled     interfaces
#        br0            8000.1c4a54447218       no              vlan1
#                                                               eth1
#                                                               eth2
#                                                               wl0.1
#                                                               wl0.2
#       
#       'vlan1' is a set of Switch ports, router on ap mode, vlan1 member include switch's Port 0 1 2 3 4 5

#       'br0' is network bridge device in linux, 'wl0.1' as it a member
#       'wl0.1' is 2.4GHZ Guest network_1, 'vlan1' and 'wl0.1' they are in same network bridge(Broadcast domain), 
#        so they can communicate with each other.
#       'eth1' is 2.4GHZ primary network
#       'eth2' is 5GHZ primary network

# exec 'robocfg show' in shell can show switch ports and vlans(switch inside only, Not on linux)

# Notice: all traffic is transport by eth0(swtich's Port 5) to linux(merlin system)

#       Switch: enabled
#       Port 0: 1000FD enabled stp: none vlan: 1 jumbo: off mac: 14:82:c4:f4:40:20
#       Port 1:   DOWN enabled stp: none vlan: 1 jumbo: off mac: 00:00:00:00:00:00
#       Port 2:   DOWN enabled stp: none vlan: 1 jumbo: off mac: 00:00:00:00:00:00
#       Port 3:   DOWN enabled stp: none vlan: 1 jumbo: off mac: 00:00:00:00:00:00
#       Port 4:   DOWN enabled stp: none vlan: 1 jumbo: off mac: 00:00:00:00:00:00
#       Port 5: 1000FD enabled stp: none vlan: 1 jumbo: off mac: 4c:2d:34:14:31:d8
#       Port 7:   DOWN enabled stp: none vlan: 1 jumbo: off mac: 00:00:00:00:00:00
#       Port 8:   DOWN enabled stp: none vlan: 1 jumbo: off mac: 00:00:00:00:00:00
#       VLANs: BCM5301x enabled mac_check mac_hash
#       1: vlan1: 0 1 2 3 4 5t
#       2: vlan2: 5t

#       On ASUS AC66U_B1 router 'Port 0' is correspond a physical Port --> WAN(blue)
#       On my asus AC66U_B1 router like below correspond physical Port
#       Port 1 --> LAN 1 
#       Port 2 --> LAN 2 
#       Port 3 --> LAN 3 
#       Port 4 --> LAN 4 
#       Port 5(eth0) is directly connected to CPU, it always UP
#     

# this setup:
#       WAN port(Port 0) will be as trunk port, transport vlan 102 traffic and vlan 200 traffic

#       'vlan 1' on Port 0 is untagged, purposes of management router
#       'vlan 101' on Port 0 is tagged, isolation primary network and Guests_1 network will use it.
#       'vlan 200' on Port 0 is tagged, isolation primary network and Guests_2 network will use it.

# client_traffic --> 2.4ghz network(wl0.1)--> br102 --> linux interface vlan102 --> switch's Port 5(tagged) -->
#  --->switch's Port 0(tagged)---->up Link Port

#       LAN ports (Port1~4) and primary WIFI will be on vlan 1
#       Guest network_1 will be on VLAN 102
#       Guest network_2 will be on VLAN 200



# Let's get started!

#!/bin/sh

# configure vlans on switch ports
# robocfg is Broadcom BCM5325/535x/536x/5311x switch configuration utility

robocfg vlan 200 ports "0t 5t"
robocfg vlan 102 ports "0t 5t"

# add vlan interface on merlin at eth0[switch 5 Port]
vconfig add eth0 200
vconfig add eth0 102

# then up it
ifconfig vlan200 up
ifconfig vlan102 up

# remove wl0.1 from br0   wl0.1-->guest network_1   wl0.2-->guest network_2
brctl delif br0 wl0.2
brctl delif br0 wl0.1

# add linux network bridge
brctl addbr br200
brctl addbr br102

# add wl0.1 and wl0.2 to linux network bridge
brctl addif br200 wl0.2
brctl addif br102 wl0.1

# add vlan102 interface and vlan200 interface to linux network bridge
brctl addif br200 vlan200
brctl addif br102 vlan102

# up linux network bridge
ifconfig br200 up
ifconfig br102 up

# setting nvram values must be correct. if NOT correct, will reject wireless client request.
nvram set br0_ifname="br0"
nvram set lan_ifname="br0"
nvram set lan_ifnames="vlan1 eth1 eth2"
nvram set br0_ifnames="vlan1 eth1 eth2"


nvram set lan1_ifnames="vlan200 wl0.2"
nvram set lan1_ifname="br200"
nvram set br200_ifname="br200"
nvram set br200_ifnames="vlan200 wl0.2"

nvram set lan2_ifnames="vlan102 wl0.1"
nvram set lan2_ifname="br102"
nvram set br102_ifname="br102"
nvram set br102_ifnames="vlan102 wl0.1"

killall eapd

eapd

# Flush ebtables --> clear all rules
ebtables -F


