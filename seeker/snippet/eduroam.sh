#date: 2024-02-07T16:54:44Z
#url: https://api.github.com/gists/da82f2e55e9fadafea0d809d3a63c3a0
#owner: https://api.github.com/users/voidbert

#!/bin/sh

nmcli con add \
  type wifi \
  con-name "eduroam" \
  ifname "XXXXXXXXX" \ # Replace with interface name
  ssid "eduroam" \
  wifi-sec.key-mgmt "wpa-eap" \
  802-1x.identity "aXXXXXX@alunos.uminho.pt" \
  # 802-1x.password "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" \ # must actually be password file
  802-1x.system-ca-certs "yes" \
  802-1x.domain-suffix-match "uminho.pt" \
  802-1x.eap "peap" \
  802-1x.phase2-auth "mschapv2"

nmcli connection up eduroam --ask