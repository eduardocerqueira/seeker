#date: 2023-11-29T17:09:30Z
#url: https://api.github.com/gists/e137eb802852cecaf544f68088cdc5cf
#owner: https://api.github.com/users/davejagoda

#!/bin/bash
# assume the number delimiter is $
cut -d '$' -f 2 /tmp/foo | paste -sd+ - | bc