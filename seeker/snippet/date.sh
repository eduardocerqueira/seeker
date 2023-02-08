#date: 2023-02-08T16:59:03Z
#url: https://api.github.com/gists/1c11479813bc5ec48fd2e5e648c43384
#owner: https://api.github.com/users/tuxfight3r

#!/bin/bash
date1="2023-02-08 10:50:33"
date2="2023-02-08 14:10:33"
date1_seconds=$(date -d "$date1" +"%s")
date2_seconds=$(date -d "$date2" +"%s")
duration=$(( $date2_seconds - $date1_seconds ))
echo "Time Elapsed: $(($duration/3600)) hours $(($duration %3600 / 60)) minutes and $(($duration % 60)) seconds."