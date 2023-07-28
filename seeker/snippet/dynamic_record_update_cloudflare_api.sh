#date: 2023-07-28T16:43:47Z
#url: https://api.github.com/gists/5fa7b252e5d2fa75fc46e8eb67f85cac
#owner: https://api.github.com/users/bestrocker221

#!/bin/bash

ZONE_ID=<YOUR_ZONE_ID>
A_record=<domain/subdomain>
A_record_id=<RECORD ID>

CF_BEARER= "**********"

LOG_FILE="./ip_log.txt"
DNS_UPDATE_LOG_FILE="./update_dns_history.log"

IP=$(curl -s https://ipinfo.io/ip)


if [[ -n "$PUBLIC_IP" ]]; then
    DATE_TIME=$(date +"%Y-%m-%d %H:%M:%S")
    LAST_IP=$(tail -n 1 "$LOG_FILE" | awk '{print $4}')
    
    if [ "$LAST_IP" != "$IP" ]; then

        DATA_TO_UPDATE="{\"type\":\"A\",\"name\":\"$A_record\",\"content\":\"$IP\",\"ttl\":1,\"proxied\":false}"
        curl -s -X PUT "https://api.cloudflare.com/client/v4/zones/"$ZONE_ID"/dns_records/"$A_record_id -H "Authorization: Bearer ${CF_BEARER}" -H "Content-Type:application/json" --data $DATA_TO_UPDATE >> ${DNS_UPDATE_LOG_FILE}
        echo "${DATE_TIME} - ${PUBLIC_IP}" >> "$LOG_FILE"
         
    fi
fi


# get records
#curl https://api.cloudflare.com/client/v4/zones/"$ZONE_ID"/dns_records -H "Authorization: Bearer ${CF_BEARER}"
RER}"
