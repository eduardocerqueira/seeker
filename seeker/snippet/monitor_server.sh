#date: 2025-07-18T16:54:19Z
#url: https://api.github.com/gists/bdf7f6da180d57e4b39bee1041db3c86
#owner: https://api.github.com/users/kakiang

#!/bin/bash

# SNMP Monitoring Script for Alpine Server
# Uses SNMPv2c with community 'public'
# Displays all metrics in human-readable format

###############################################################################
# CONFIGURATION
###############################################################################

# Set the target host (localhost)
TARGET="127.0.0.1"

# Set SNMP community string
COMMUNITY="public"

# Set colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

###############################################################################
# SYSTEM INFORMATION
###############################################################################

# Get system description
echo -e "${GREEN}=== System Information ===${NC}"
sysDescr=$(snmpget -v 2c -c $COMMUNITY $TARGET .1.3.6.1.2.1.1.1.0 | awk -F'"' '{print $2}')
echo "System OS   : $sysDescr"

# Get uptime
sysUptime=$(snmpget -v 2c -c $COMMUNITY $TARGET .1.3.6.1.2.1.1.3.0 | awk '{print $4}' | sed 's/.\(..\)/ days \1 hours/')
echo "Uptime      : $sysUptime"

# Get contact info
sysContact=$(snmpget -v 2c -c $COMMUNITY $TARGET .1.3.6.1.2.1.1.4.0 | awk -F'"' '{print $2}')
echo "Admin Contact: $sysContact"

###############################################################################
# CPU & MEMORY
###############################################################################

echo -e "\n${GREEN}=== CPU & Memory ===${NC}"

# Get CPU load averages
load1=$(snmpget -v 2c -c $COMMUNITY $TARGET .1.3.6.1.4.1.2021.10.1.3.1 | awk '{print $4}')
load5=$(snmpget -v 2c -c $COMMUNITY $TARGET .1.3.6.1.4.1.2021.10.1.3.2 | awk '{print $4}')
load15=$(snmpget -v 2c -c $COMMUNITY $TARGET .1.3.6.1.4.1.2021.10.1.3.3 | awk '{print $4}')
echo "CPU Load    : 1min=${load1}, 5min=${load5}, 15min=${load15}"

# Get CPU usage percentage
cpuUsage=$(snmpget -v 2c -c $COMMUNITY $TARGET NET-SNMP-EXTEND-MIB::nsExtendOutput1.\"cpu-usage\" | awk -F'"' '{print $2}')
echo "CPU Usage   : $cpuUsage"

# Get memory usage
memTotal=$(snmpget -v 2c -c $COMMUNITY $TARGET .1.3.6.1.4.1.2021.4.5.0 | awk '{print $4}')
memFree=$(snmpget -v 2c -c $COMMUNITY $TARGET .1.3.6.1.4.1.2021.4.6.0 | awk '{print $4}')
memUsed=$((memTotal - memFree))
memPercent=$((memUsed*100/memTotal))
echo "Memory Usage: ${memUsed}KB/${memTotal}KB (${memPercent}%)"

###############################################################################
# DISK USAGE
###############################################################################

echo -e "\n${GREEN}=== Disk Usage ===${NC}"

# Get root partition usage
rootUsage=$(snmpget -v 2c -c $COMMUNITY $TARGET .1.3.6.1.4.1.2021.9.1.9.1 | awk '{print $4}')
echo "Root Partition: ${rootUsage}% used"

# Get /var partition usage
varUsage=$(snmpget -v 2c -c $COMMUNITY $TARGET .1.3.6.1.4.1.2021.9.1.9.2 | awk '{print $4}')
echo "Var Partition : ${varUsage}% used"

###############################################################################
# NETWORK & SERVICES
###############################################################################

echo -e "\n${GREEN}=== Network & Services ===${NC}"

# Check IP forwarding status
ipForward=$(snmpget -v 2c -c $COMMUNITY $TARGET NET-SNMP-EXTEND-MIB::nsExtendOutput1.\"ip-forwarding\" | awk -F'"' '{print $2}')
[ "$ipForward" -eq 1 ] && status="${GREEN}Enabled${NC}" || status="${RED}Disabled${NC}"
echo "IP Forwarding: $status"

# Check DNS server status
dnsStatus=$(snmpget -v 2c -c $COMMUNITY $TARGET .1.3.6.1.4.1.2021.2.1.100.1 | awk '{print $4}')
[ "$dnsStatus" -eq 1 ] && status="${GREEN}Running${NC}" || status="${RED}Stopped${NC}"
echo "DNS Service  : $status"

# Check DHCP server status
dhcpStatus=$(snmpget -v 2c -c $COMMUNITY $TARGET .1.3.6.1.4.1.2021.2.1.100.2 | awk '{print $4}')
[ "$dhcpStatus" -eq 1 ] && status="${GREEN}Running${NC}" || status="${RED}Stopped${NC}"
echo "DHCP Service : $status"

# Get active DHCP leases
dhcpLeases=$(snmpget -v 2c -c $COMMUNITY $TARGET NET-SNMP-EXTEND-MIB::nsExtendOutput1.\"dhcp-leases\" | awk -F'"' '{print $2}')
echo "DHCP Leases  : $dhcpLeases active"

###############################################################################
# NAT CONNECTIONS
###############################################################################

echo -e "\n${GREEN}=== NAT Statistics ===${NC}"

# Get active NAT connections
natConns=$(snmpget -v 2c -c $COMMUNITY $TARGET NET-SNMP-EXTEND-MIB::nsExtendOutput1.\"nat-connections\" | awk -F'"' '{print $2}')
echo "Active NAT Connections: $natConns"

###############################################################################
# FINAL STATUS CHECK
###############################################################################

echo -e "\n${GREEN}=== Overall Status ===${NC}"

# Check if any critical service is down
if [ "$dnsStatus" -ne 1 ] || [ "$dhcpStatus" -ne 1 ]; then
    echo -e "${RED}CRITICAL: Essential services down!${NC}"
else
    echo -e "${GREEN}All essential services operational${NC}"
fi

# Check disk space warnings
if [ "$rootUsage" -gt 85 ] || [ "$varUsage" -gt 85 ]; then
    echo -e "${YELLOW}WARNING: Disk space running low${NC}"
fi