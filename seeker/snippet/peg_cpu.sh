#date: 2022-06-23T16:59:20Z
#url: https://api.github.com/gists/669ad1f1b21b482649a823cfa9581774
#owner: https://api.github.com/users/evu

#!/bin/bash
for i in $(seq $(getconf _NPROCESSORS_ONLN)); do yes > /dev/null & done

echo "To stop, run:"
echo "killall yes"
echo

sleep 60
killall yes