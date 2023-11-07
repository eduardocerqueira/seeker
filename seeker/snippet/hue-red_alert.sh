#date: 2023-11-07T17:06:58Z
#url: https://api.github.com/gists/82460273b5e98626907560a49925891a
#owner: https://api.github.com/users/cmj

#!/usr/bin/env bash
# pulse Hue™ light for notifications/alerts
#
# example to find key (username) using 'luminance'
# https://github.com/BOSSoNe0013/luminance/

# $ dconf read /com/craigcabrey/luminance/username
# '111111FFFFFF00000007HUj7rBFi5vrSKDw99999'

HUE_KEY=''
LIGHT=3
SAT=254 # saturation
DIM=20
BRIGHT=255
HUE=255 # red corner light

for((i=1;i<=4;i++)); do 
  curl -XPUT -H 'Content-Type: application/json' \
    http://huehub/api/$HUE_KEY/lights/$LIGHT/state \
    -d "{\"on\":true,\"sat\":$SAT,\"bri\":$DIM,\"hue\":$HUE}" \
    -so /dev/null
  sleep .5
  curl -XPUT -H 'Content-Type: application/json' \
    http://huehub/api/$HUE_KEY/lights/$LIGHT/state \
    -d "{\"on\":true,\"sat\":$SAT,\"bri\":$BRIGHT,\"hue\":$HUE}" \
    -so /dev/null
  sleep .5
done

# Turn light off 
#curl -XPUT -H 'Content-Type: application/json' http://huehub/api/$HUE_KEY/lights/$LIGHT/state -d '{"on":false}' -so /dev/null
