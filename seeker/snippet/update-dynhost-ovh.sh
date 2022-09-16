#date: 2022-09-16T22:07:45Z
#url: https://api.github.com/gists/3951195b568e004533a1ff3b9ae63a34
#owner: https://api.github.com/users/ps1dr3x

#!/bin/bash

# Don't forget to chmod +x update-dynhost-ovh.sh and add the cron
# */5 * * * * update-dynhost-ovh.sh

DOMAIN=""
USERNAME=""
PASSWORD= "**********"

CURRENT_IP=$(curl ipinfo.io/ip)
CURRENT_IP_DNS=$(dig +short $DOMAIN)

UPDATE_RESULT=""
if [ "$CURRENT_IP_DNS" != "$CURRENT_IP" ]; then
    UPDATE_RESULT=$(curl "https: "**********":$PASSWORD")
else
    UPDATE_RESULT="The DNS is already up to date."
fi

TIMESTAMP=$(date +%s)

echo "$TIMESTAMP: $UPDATE_RESULT" >> /var/log/update-dyndns-ovh.logate-dyndns-ovh.log
TE_RESULT" >> /var/log/update-dyndns-ovh.logate-dyndns-ovh.log
