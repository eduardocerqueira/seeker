#date: 2022-07-04T17:00:41Z
#url: https://api.github.com/gists/c5a5f422f3d79a196f69f920900d1544
#owner: https://api.github.com/users/wandersonlima

#!/bin/bash
for vol in $(heketi-cli volume list | awk '{print $1}' | cut -d':' -f2)
do
heketi-cli volume delete $vol
done