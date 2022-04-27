#date: 2022-04-27T17:00:44Z
#url: https://api.github.com/gists/abfae7b978c7c8bf9a00ef48558bdd99
#owner: https://api.github.com/users/kiloutyg

#!/bin/bash

# A bash script to update a Cloudflare DNS A record with the external IP of the source machine
# Used to provide DDNS service for my home
# Needs the DNS record pre-creating on Cloudflare

## Based on https://gist.github.com/Tras2/cba88201b17d765ec065ccbedfb16d9a with updates to use
## per-zone configurable access tokens available in the API sections of your Cloudflare profile
##  - info@foo-games.com

# Proxy - uncomment and provide details if using a proxy
#export https_proxy=http://<proxyuser>:<proxypassword>@<proxyip>:<proxyport>

# Cloudflare zone is the zone which holds the record
dnsrecord=example.com
zoneid=examplezoneidhashfromdomainoverviewpage

## Cloudflare authentication details
## keep these private
cloudflare_auth_key=authkeycreatedthroughnewbetaapitokeninterface

# Get the current external IP address
ip=$(curl -s -X GET https://checkip.amazonaws.com)

#echo "Current IP is $ip"

if host $dnsrecord 1.1.1.1 | grep "has address" | grep "$ip"; then
#  echo "$dnsrecord is currently set to $ip; no changes needed"
  exit
fi

# if here, the dns record needs updating

# get the dns record id
dnsrecordid=$(curl -v -s -X GET "https://api.cloudflare.com/client/v4/zones/$zoneid/dns_records?type=A&name=$dnsrecord" \
  -H "Authorization: Bearer $cloudflare_auth_key" \
  -H "Content-Type: application/json" | jq -r  '{"result"}[] | .[0] | .id')

# update the record
curl -s -X PUT "https://api.cloudflare.com/client/v4/zones/$zoneid/dns_records/$dnsrecordid" \
  -H "Authorization: Bearer $cloudflare_auth_key" \
  -H "Content-Type: application/json" \
  --data "{\"type\":\"A\",\"name\":\"$dnsrecord\",\"content\":\"$ip\",\"ttl\":1,\"proxied\":false}" | jq
