#date: 2021-10-21T17:17:33Z
#url: https://api.github.com/gists/8df19bcaf075ca573ce636dc4cbd2c7d
#owner: https://api.github.com/users/bakkeby

#!/bin/sh
#COLOR=^c#FFF7D4^
TIME="$(date '+%B %dXX' | sed -r -e 's/(1[123])XX/\1th/;s/1XX/1st/;s/2XX/2nd/;s/3XX/3rd/;s/XX/th/;s/ 0/ /'), $(date +'%H:%M:%S')"
echo "${COLOR}${TIME}^d^"