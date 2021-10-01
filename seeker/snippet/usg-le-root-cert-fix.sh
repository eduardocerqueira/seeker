#date: 2021-10-01T01:33:45Z
#url: https://api.github.com/gists/aeff3e367c77b2b01ac8c0ea30491c9d
#owner: https://api.github.com/users/sprocktech

#!/bin/sh
sed -i '/DST_Root_CA_X3.crt/d' /etc/ca-certificates.conf
curl -sk https://letsencrypt.org/certs/isrgrootx1.pem -o /usr/local/share/ca-certificates/ISRG_Root_X1.crt
curl -sk https://letsencrypt.org/certs/isrg-root-x2.pem -o /usr/local/share/ca-certificates/ISRG_Root_X2.crt
curl -sk https://letsencrypt.org/certs/lets-encrypt-r3.pem -o /usr/local/share/ca-certificates/Lets_Encrypt_R3.crt
update-ca-certificates --fresh