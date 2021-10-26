#date: 2021-10-26T16:54:41Z
#url: https://api.github.com/gists/e2902dee36d78ac65dcab83738c60c5f
#owner: https://api.github.com/users/krcs

#!/bin/sh

opkg update

for line in $(opkg list-upgradable | awk 'BEGIN { FS=" " }; { print $1  }')
do
   opkg upgrade $line
done;