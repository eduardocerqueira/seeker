#date: 2025-04-03T17:13:44Z
#url: https://api.github.com/gists/a9c2f5e32746d3c04ffd1373228fcce4
#owner: https://api.github.com/users/PikkonMG

#!/bin/bash

# Polls AT&T's DHCP server for updates, to keep static IPs alive.
# This allows UDM Pro users to set their DHCP IP as 'static' in the 'Internet' section
# allowing the use of static IP configuration in Unifi Network.
# 1. Find your DHCP IP.
# 2. Set Internet IPv4 to Static IP, and enter your DHCP address. Gateway is going to be .1
# 3. Add your static IP block to Additional IP Addresses
# 4. Place this script in the on_boot.d/ directory: https://github.com/unifi-utilities/unifios-utilities/tree/main/on-boot-script
# 5. After reboot, check the script is working: cat /var/log/udhcpc.log
# Credit to https://community.ui.com/questions/Additional-IP-with-DHCP-primary-on-UDM-Pro/ceeaa11b-b1f2-442d-a8ba-6cdfcc29c7f6
# Tested on 3.0.20

PUBLIC_DHCP_IP=""
# eth8 is RJ45, eth9 is SFP+ on UDMP
WAN_PORT="eth9"

nohup /usr/bin/busybox-legacy/udhcpc --foreground --interface $WAN_PORT --script /usr/share/ubios-udapi-server/ubios-udhcpc-script -r $PUBLIC_DHCP_IP >/var/log/udhcpc.log 2>&1 &