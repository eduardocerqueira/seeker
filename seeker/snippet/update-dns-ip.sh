#date: 2022-06-06T17:07:19Z
#url: https://api.github.com/gists/866db82e0d84c73c73d4bf824e9aac8c
#owner: https://api.github.com/users/rekyuu

#! /bin/bash

DNS_NAME=""
ZONE_ID=""
API_TOKEN=""
DNS_RECORD_ID=""

log-to-file() {
    TIMESTAMP=$(TZ="America/Boise" date --iso-8601="seconds")
    echo "[$TIMESTAMP] $1" >> update-dns-ip.log
}

REMOTE_IP=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records/$DNS_RECORD_ID" \
    -H "Authorization: Bearer $API_TOKEN" \
    -H "Content-Type:application/json" | jq -r ".result.content")

LOCAL_IP=$(curl -s http://checkip.amazonaws.com)

if [[ "$REMOTE_IP" != "$LOCAL_IP" ]]; then
    echo "IP changed from $REMOTE_IP to $LOCAL_IP. Updating with CloudFlare."

    log-to-file "$REMOTE_IP -> $LOCAL_IP"

    UPDATE_RESULT=$(curl -s -X PUT "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records/$DNS_RECORD_ID" \
        -H "Authorization: Bearer $API_TOKEN" \
        -H "Content-Type: application/json" \
        --data "{\"type\":\"A\",\"name\":\"$DNS_NAME\",\"content\":\"$LOCAL_IP\",\"ttl\":1,\"proxied\":false}")
        
    log-to-file "$UPDATE_RESULT"
fi