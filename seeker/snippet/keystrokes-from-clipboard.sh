#date: 2022-02-14T17:10:37Z
#url: https://api.github.com/gists/c555c42eb2ee14f202d5bd7134e7cce6
#owner: https://api.github.com/users/netzverweigerer

#!/bin/sh
xclip -selection clipboard -out | tr \\n \\r | xdotool selectwindow windowfocus type --clearmodifiers --delay 25 --window %@ --file -

