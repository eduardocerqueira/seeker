#date: 2022-04-14T16:59:24Z
#url: https://api.github.com/gists/c74b395ed046e38a730776fc73e93e83
#owner: https://api.github.com/users/mothdotmonster

#!/bin/bash

PICDIR='/home/chesapeake/Pictures/wp/desktop' # change this

FILE=$(ls -t "$PICDIR" | head -1)

feh --bg-fill "$PICDIR"/"$FILE"
