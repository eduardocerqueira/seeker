#date: 2024-08-15T16:35:22Z
#url: https://api.github.com/gists/2538f79f4bea37748b97a7903872c1cc
#owner: https://api.github.com/users/lytithwyn

#!/bin/sh

PATH=/usr/bin:/usr/sbin:/bin:/sbin
BL_FILE="/tmp/oisd_nsfw_dnsmasq2.txt"
NL='
'

sleep 20

# check if the blacklist already exists
if [ -f "${BL_FILE}" ]; then
    exit 0
fi

# download the block list to /tmp
curl -o "${BL_FILE}" -L https://nsfw.oisd.nl/dnsmasq2

# if we failed, bail out
if [ "$?" != "0" ]; then
    exit 1
fi

# we succeeded - capture the current dnsmasq options and append our blacklist
DNSMASQ_CONF=$(nvram get dnsmasq_options)
DNSMASQ_CONF_MOD="${DNSMASQ_CONF}${NL}conf-file=${BL_FILE}"

# set the dnsmasq_options nvram variable to the copy that has the blacklist
nvram set dnsmasq_options="${DNSMASQ_CONF_MOD}"

# reload dnsmasq to get it to pick that up
service dnsmasq restart

# reset our dnsmasq_options back to the original in case someone does a 'commit'
# if our blacklist is specified in the options on boot and doesn't exist (which it won't)
# dnsmasq will crash
nvram set dnsmasq_options="${DNSMASQ_CONF}"

exit 0