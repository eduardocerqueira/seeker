#date: 2021-10-11T17:12:51Z
#url: https://api.github.com/gists/3bdfab948c61f9968f11db124aeac8cd
#owner: https://api.github.com/users/aardbol

#!/bin/bash

# Assuming you've configured your nginx config:
# ssl_certificate /etc/letsencrypt/live/[domain]/fullchain.pem;
# ssl_certificate_key /etc/letsencrypt/live/[domain]/privkey.pem;
#
# Change the variables between [] and add your OVH API keypair

export OVH_AK=""
export OVH_AS=""

# /home/[username]/.acme.sh/[domain]
SRC=""
# /etc/letsencrypt/live/[domain]
DEST=""

/home/[username]/.acme.sh/acme.sh --issue --force -d [domain] -d [domain] --dns dns_ovh && \
sudo mv $SRC/[domain].key $DEST/privkey.pem && \
sudo cp $SRC/fullchain.cer $DEST/fullchain.pem && \
sudo chown root:root $DEST/privkey.pem && \
sudo chmod 600 $DEST/privkey.pem && \
sudo systemctl restart nginx
