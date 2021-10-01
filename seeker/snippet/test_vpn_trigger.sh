#date: 2021-10-01T17:11:50Z
#url: https://api.github.com/gists/ac05386e1af049501433f117262bdd59
#owner: https://api.github.com/users/DonRichards

#!/bin/bash

:`
Only works in Linux
Replace private_url.example.com with a URL that is only accessible when on the VPN.
Replace office_vpn with the VPN name you gave on your local machine. When you created
the VPN client connection to the office you gave it a name. To see the names of your
VPN connections run this:

$ nmcli connection
NAME                UUID                                  TYPE      DEVICE
office_vpn        d28603ef-ee7c-4bba-8525-6348bcebfeec    vpn        --
`

# Connect to VPN if not already connected.
if [[ $(uname -s | tr A-Z a-z) == "linux" ]]; then
    if [[ $(nmcli connection | grep vpn | grep -o '^\S*' | grep 'office_vpn') == "office_vpn" ]]; then
        if [[ $(nmcli -f GENERAL.STATE con show office_vpn | sed 's/.*\s//') != "activated" ]]; then
            nmcli con up id $(nmcli connection | grep vpn | grep -o '^\S*' | grep 'office_vpn');
        fi
    fi
fi
echo -e "\033[0;33mChecking VPN\033[0m connection"
# While the connection is establishing this waits for it.
while ! $(ping -c 1 -n -w 1 private_url.example.com &> /dev/null); do sleep 1 && echo -ne "Please connect to the VPN, I'll wait... \033[0K\r" ; done
echo -e "\n\n\033[0;32mDone\033[0m\n\n"

echo "Sleeping for 5 seconds"
sleep 5

# Disonnect from VPN if not already disconnected.
if [[ $(uname -s | tr A-Z a-z) == "linux" ]]; then
    if [[ $(nmcli connection | grep vpn | grep -o '^\S*' | grep 'office_vpn') == "office_vpn" ]]; then
        if [[ $(nmcli -f GENERAL.STATE con show office_vpn | sed 's/.*\s//') == "activated" ]]; then
            nmcli con down id $(nmcli connection | grep vpn | grep -o '^\S*' | grep 'office_vpn');
        fi
    fi
fi
echo -e "\033[0;33mChecking VPN\033[0m connection"
# While the connection is being reset this waits.
while $(ping -c 1 -n -w 1 private_url.example.com &> /dev/null); do sleep 1 && echo -ne "Waiting while VPN disconnects... \033[0K\r" ; done
echo -e "\n\n\033[0;32mDone\033[0m\n\n"

:` Makefile example
OS_NAME := $(shell uname -s | tr A-Z a-z)
os:
	@echo $(OS_NAME)

ifeq ($(OS_NAME),linux)
ifeq ($(shell nmcli connection | grep vpn | grep -o '^\S*' | grep 'office_vpn'),office_vpn)
ifneq ($(shell nmcli -f GENERAL.STATE con show office_vpn | sed 's/.*\s//'),activated)
	@echo "Linux"
	if ping -c1 private_url.example.com >/dev/null 2>&1; then : echo "Connected!"; else nmcli con up id office_vpn ; fi
endif
endif
endif
	echo -n "\n\n\n\n Please connect to VPN, I'll wait... \n\n\n\n"
	until ping -c1 private_url.example.com >/dev/null 2>&1; do : echo 'Waiting for VPN connection'; done
`