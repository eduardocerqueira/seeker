#date: 2024-05-13T17:07:26Z
#url: https://api.github.com/gists/12767a6d25620748d69d106fd1293787
#owner: https://api.github.com/users/jwhb

#!/bin/sh
adb shell pm path com.example.myapp | sed 's/^package://g' | xargs -L1 adb pull
