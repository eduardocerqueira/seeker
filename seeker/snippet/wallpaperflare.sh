#date: 2022-04-06T17:09:03Z
#url: https://api.github.com/gists/7db40c3b93e5c4cae52b813479918015
#owner: https://api.github.com/users/whatnik

#!/bin/bash

curl -s "https://www.wallpaperflare.com/index.php?c=main&m=portal_loadmore&page=$(shuf -i 1-50 -n 1)" \
  -H 'authority: www.wallpaperflare.com' \
  -H 'accept: */*' \
  -H 'referer: https://www.wallpaperflare.com/' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-fetch-dest: empty' \
  -H 'sec-fetch-mode: cors' \
  -H 'sec-fetch-site: same-origin' \
  -H 'x-requested-with: XMLHttpRequest' \
  --compressed | grep -Eo "(https)://[a-zA-Z0-9./?=_-]*" | grep 'www' | shuf | head -1 | xargs -i curl -s "{}/download" | grep -Eo "(https)://[a-zA-Z0-9./?=_-]*" | grep '\.jpg' | head -2 | tail -1
