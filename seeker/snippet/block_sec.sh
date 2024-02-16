#date: 2024-02-16T16:49:42Z
#url: https://api.github.com/gists/acdc60a534d451742a398bddea81fba5
#owner: https://api.github.com/users/nullenc0de

#!/bin/bash

apt install golang -y
GOROOT="/usr/local/go"
PATH="${PATH}:${GOROOT}/bin"
GOPATH=$HOME/go
PATH="${PATH}:${GOROOT}/bin:${GOPATH}/bin"

go install github.com/projectdiscovery/asnmap/cmd/asnmap@latest

# Define an array of security, technology, and additional company names
companies=(
    "Symantec"
    "McAfee"
    "Trend Micro"
    "Palo Alto Networks"
    "Cisco Systems"
    "Check Point Software"
    "Fortinet"
    "FireEye"
    "CrowdStrike"
    "Proofpoint"
    "Sophos"
    "Kaspersky Lab"
    "Bitdefender"
    "Rapid7"
    "F-Secure"
    "IBM Security"
    "SentinelOne"
    "Carbon Black"
    "SonicWall"
    "Zscaler"
    "Google"
    "Microsoft"
    "Apple"
    "Amazon"
    "Facebook"
    "IBM"
    "Oracle"
    "Intel"
    "Cisco"
    "HP"
    "Dell"
    "VMware"
    "Adobe"
    "Salesforce"
    "Nvidia"
    "Qualcomm"
    "Samsung"
    "Cloudflare"
    "Avast"
)

# Loop through each company name
for company in "${companies[@]}"; do
    # Use asnmap to search for IP addresses associated with the company name
    ips=$(asnmap -org "$company" -silent)

    # Loop through each IP address found
    while IFS= read -r ip; do
        # Add the IP address to IP tables for dropping incoming packets
        iptables -A INPUT -s "$ip" -j DROP
    done <<< "$ips"
done