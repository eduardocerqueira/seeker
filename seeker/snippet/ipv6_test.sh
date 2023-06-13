#date: 2023-06-13T16:55:06Z
#url: https://api.github.com/gists/b3af44bee7fbc0d45726efc700aafa88
#owner: https://api.github.com/users/MohamedElashri

#!/bin/bash
# Usage ./ipv6_test.sh example.com
# Or to add whois ./ipv6_test.sh example.com --whois

# Check if the website URL is provided as an argument
if [ $# -eq 0 ]; then
  echo "Please provide the website URL as an argument."
  echo "Example: ./ipv6_test.sh example.com [--whois]"
  exit 1
fi

website="$1"
do_whois=false

# Check if the optional --whois argument is provided
if [ "$2" = "--whois" ]; then
  do_whois=true
fi

# Check if the website is accessible
if curl --head --silent --fail $website >/dev/null; then
  echo -e "\e[32mWebsite ($website) is accessible.\e[0m"
else
  echo -e "\e[31mWebsite ($website) is not accessible or there is an outage.\e[0m"
  exit 1
fi

# Check if the website is accessible using IPv6
if ping6 -c 1 $website >/dev/null 2>&1; then
  echo -e "\e[32mWebsite ($website) is accessible using IPv6.\e[0m"
else
  echo -e "\e[31mWebsite ($website) is not accessible using IPv6.\e[0m"
fi

# Check DNS resolution and retrieve IPv6 address
ipv6_address=$(dig +short AAAA $website | head -n 1)

if [ -n "$ipv6_address" ]; then
  echo -e "\e[32mDNS resolution successful. IPv6 address: $ipv6_address\e[0m"
else
  echo -e "\e[31mDNS resolution failed. No valid IPv6 address found.\e[0m"
fi

# WHOIS information
if [ "$do_whois" = true ]; then
  echo -e "\n\e[33mWHOIS Information:\e[0m"
  whois $website
fi

# Traceroute
traceroute_output=$(traceroute -6 -q 1 $website 2>/dev/null)

if [ -n "$traceroute_output" ]; then
  echo -e "\n\e[33mTraceroute Output:\n$traceroute_output\e[0m"
else
  echo -e "\n\e[33mTraceroute information not available.\e[0m"
fi
