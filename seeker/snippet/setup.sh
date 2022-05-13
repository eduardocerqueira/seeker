#date: 2022-05-13T17:14:55Z
#url: https://api.github.com/gists/6b5e9d434128a045a4fd1082b58447ba
#owner: https://api.github.com/users/devorbitus

#!/usr/bin/env bash

# ---------------------------------------------------------------
# ------- Input Variables -----------------------BEGIN-----------
# ---------------------------------------------------------------
# Enter the access ID to be used for authentication
ADMIN_ACCESS_ID=""
# Enter the access key to be used for authentication
ADMIN_ACCESS_KEY=""
# Enter the access ID for the Single Sign On (SAML/OIDC)
SSO_ACCESS_ID=""
# ---------------------------------------------------------------
# ------- Input Variables -----------------------END-------------
# ---------------------------------------------------------------

if [ "$ADMIN_ACCESS_ID" == "" ]; then
    echo "Error: Required ADMIN_ACCESS_ID environment variable not set. Exiting..."
    exit 1
fi

if [ "$ADMIN_ACCESS_KEY" == "" ]; then
    echo "Error: Required ADMIN_ACCESS_KEY environment variable not set. Exiting..."
    exit 1
fi

if [ "$SSO_ACCESS_ID" == "" ]; then
    echo "Error: Required SSO_ACCESS_ID environment variable not set. Exiting..."
    exit 1
fi

cat << EOF >| $PWD/docker-compose.yml
version: '3.3'
services:
  agw:
    environment:
      - ALLOWED_ACCESS_IDS=$SSO_ACCESS_ID
      - ADMIN_ACCESS_ID=$ADMIN_ACCESS_ID
      - ADMIN_ACCESS_KEY=$ADMIN_ACCESS_KEY
      - CLUSTER_NAME=dockerCompose
      #- http_proxy=''
      #- https_proxy=''
      #- no_proxy=''
    ports:
      - '8000:8000'
      - '8200:8200'
      - '18888:18888'
      - '8080:8080'
      - '8081:8081'
    container_name: agw
    restart: unless-stopped
    image: akeyless/base

EOF

echo -e "Command to start docker-compose:\ndocker-compose up -d"
echo -e "Command to scale to multiple instances:\ndocker-compose up -d --scale agw=2"
echo -e "\nCommand to view Gateway logs:\ndocker logs -f agw"
echo -e "\nCommand to exec into the Gateway:\ndocker exec -it agw bash"
