#date: 2025-06-06T17:05:16Z
#url: https://api.github.com/gists/7d479178b89aad7b539b97c65b4c3ffd
#owner: https://api.github.com/users/sunipkm

nmcli con add \
  type wifi \
  con-name "eduroam"
  ifname "wlp4s0" \ # Your wifi interface
  ssid "eduroam" \
  wifi-sec.key-mgmt "wpa-eap" \
  802-1x.identity "<YOUR-STUDENT-ID>@lu.se" \ # May also use another university identification
  802-1x.password "<YOUR-PASSWORD" \
  802-1x.system-ca-certs "yes" \
  802-1x.domain-suffix-match "radius.lu.se" \
  802-1x.eap "peap" \
  802-1x.phase2-auth "mschapv2"
  
nmcli connection up eduroam --ask # ask for password