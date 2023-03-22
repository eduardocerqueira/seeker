#date: 2023-03-22T17:10:26Z
#url: https://api.github.com/gists/5b45e37611010c68c089bbb256ac4c67
#owner: https://api.github.com/users/roberts91

#!/bin/bash

# A bash script to update a Cloudflare DNS A record with the external IP of the source machine
# Used to provide DDNS service for my home
# Needs the DNS record pre-creating on Cloudflare

# Proxy - uncomment and provide details if using a proxy
#export https_proxy=http: "**********":<proxypassword>@<proxyip>:<proxyport>

# Cloudflare zone is the zone which holds the record
zoneid=zone-id
#zone=domain.com

# dnsrecord is the A record which will be updated, comma separated
dnsrecords=(
    "domain.com"
    "domain.eu"
)

## Cloudflare authentication details
cloudflare_auth_key="api-key"

# Get the current external IP address
ip=$(curl -s -X GET https://checkip.amazonaws.com)

echo "Current IP is $ip"

# if here, the dns record needs updating
# get the zone id for the requested zone
#zoneid=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones?name=$zone&status=active" \
#  -H "Authorization: Bearer $cloudflare_auth_key" \
#  -H "Content-Type: application/json" | jq -r '{"result"}[] | .[0] | .id')
#echo "Zoneid for $zone is $zoneid"
for dnsrecord in "${dnsrecords[@]}"
do

if host $dnsrecord 1.1.1.1 | grep "has address" | grep "$ip"; then
  echo "$dnsrecord is currently set to $ip; no changes needed"
  continue
fi

# get the dns record id
dnsrecordid=$(curl -s -X GET "https://api.cloudflare.com/client/v4/zones/$zoneid/dns_records?type=A&name=$dnsrecord" \
  -H "Authorization: Bearer $cloudflare_auth_key" \
  -H "Content-Type: application/json" | jq -r '{"result"}[] | .[0] | .id')

echo "DNSrecordid for $dnsrecord is $dnsrecordid"

# update the record
curl -s -X PUT "https://api.cloudflare.com/client/v4/zones/$zoneid/dns_records/$dnsrecordid" \
  -H "Authorization: Bearer $cloudflare_auth_key" \
  -H "Content-Type: application/json" \
  --data "{\"type\":\"A\",\"name\":\"$dnsrecord\",\"content\":\"$ip\",\"ttl\":1,\"proxied\":false}" | jq
done