#date: 2025-06-25T17:15:15Z
#url: https://api.github.com/gists/50d187480bdec35528ab2a8c7f7e995d
#owner: https://api.github.com/users/sofianhw

#!/bin/sh
adb devices | tail -n +2 | cut -sf 1 | xargs -I \{\} -P4 adb -s \{\} install -r $1