#date: 2021-12-02T16:59:35Z
#url: https://api.github.com/gists/a121eb4482eb814eb20eb692234b65dc
#owner: https://api.github.com/users/JonasGroeger

#!/usr/bin/env sh

HTTP_PROXY_HOST=localhost
HTTP_PROXY_PORT=9000

HTTPS_PROXY_HOST=localhost
HTTPS_PROXY_PORT=9000

jshell \
  proxy.jsh

jshell \
  -R-Dhttp.proxyHost="$HTTP_PROXY_HOST" \
  -R-Dhttp.proxyPort="$HTTP_PROXY_PORT" \
  -R-Dhttps.proxyHost="$HTTPS_PROXY_HOST" \
  -R-Dhttps.proxyPort="$HTTPS_PROXY_PORT" \
  proxy.jsh
