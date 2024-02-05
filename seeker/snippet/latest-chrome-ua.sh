#date: 2024-02-05T16:55:21Z
#url: https://api.github.com/gists/57443972d1284fdd862e0953706b30b8
#owner: https://api.github.com/users/s3rgeym

#!/bin/bash

CACHE_TIME=3600
LATEST_CHROME_UA=/tmp/latest-chrome-ua.txt
if [ ! -f "$LATEST_CHROME_UA" ] || [ $(($(date +%s) - $(stat -c %Y "$LATEST_CHROME_UA"))) -ge $CACHE_TIME ]; then
  echo "retrieve latest chrome user agent" >&2
  curl -s https://www.whatismybrowser.com/guides/the-latest-user-agent/chrome | grep -oP '(?<=<li><span class="code">)[^<]+' | head -1 >"$LATEST_CHROME_UA"
fi
cat "$LATEST_CHROME_UA"
