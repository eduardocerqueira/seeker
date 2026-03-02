#date: 2026-03-02T17:24:54Z
#url: https://api.github.com/gists/963ca9d0af74e85982400490d4589fe4
#owner: https://api.github.com/users/cmbaughman

#!/bin/bash

# INSTALL
# =======
# Add Cloudflare's gpg key
curl -fsSL https://pkg.cloudflareclient.com/pubkey.gpg | sudo gpg --yes --dearmor --output /usr/share/keyrings/cloudflare-warp-archive-keyring.gpg

# Add the repository
echo "deb [signed-by=/usr/share/keyrings/cloudflare-warp-archive-keyring.gpg] https://pkg.cloudflareclient.com/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/cloudflare-client.list

# Install
sudo apt update && sudo apt install cloudflare-warp

## CONFIG/USAGE
# =============
# Set to use SOCKS5 mode
warp-cli mode proxy
warp-cli connect

# Making wget use the proxy
wget -e use_proxy=yes -e http_proxy=127.0.0.1:40000 http://filtered_domain.com

## THAT'S PRETTY MUCH IT. HAVE YOUR SCRIPT TURN THAT ON AND MAKE ALL REQUESTS WITH THIS PROXY.
## 
## YOU CAN ALSO USE FLAREPROX - https://github.com/MrTurvey/flareprox AND KEEP CHANGING YOUR IP
## FOR EVERY REQUEST.
