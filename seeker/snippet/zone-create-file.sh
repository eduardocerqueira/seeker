#date: 2023-11-28T17:07:00Z
#url: https://api.github.com/gists/b759860fb692c7e18033a3497d86d029
#owner: https://api.github.com/users/imrishabh18

#!/bin/bash

# Prompt user for domain name
read -p "Enter the domain name: " DOMAIN
read -p "AAAA record: " AAAA
read -p "A record: " A
read -p "TXT record: " TXT

ZONE_FILE="/etc/bind/db.${DOMAIN}"

# Create the zone file
cat > ${ZONE_FILE} <<EOF
\$TTL 86400
@ IN SOA ${DOMAIN}. abuse.${DOMAIN}. (
              2023110101 ; Serial
              3600       ; Refresh
              1800       ; Retry
              604800     ; Expire
              86400 )    ; Minimum TTL

@ IN NS ns1.${DOMAIN}.
ns1 IN A ${A}
@ IN A ${A}
@ IN AAAA ${AAAA}
www IN A ${A}
@ IN RP abuse.${DOMAIN} *.${DOMAIN} 
* IN TXT ${TXT}

EOF

# Update DNS server configuration
echo "zone \"${DOMAIN}\" {" >> /etc/bind/zones.rfc1918
echo "    type master;" >> /etc/bind/zones.rfc1918
echo "    file \"/etc/bind/db.${DOMAIN}\";" >> /etc/bind/zones.rfc1918
echo "};" >> /etc/bind/zones.rfc1918

# Reload DNS server
systemctl restart bind9

echo "Zone file created for ${DOMAIN}."
