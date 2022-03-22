#date: 2022-03-22T17:17:05Z
#url: https://api.github.com/gists/38a045d17a4b5d94bda1c93e7db56444
#owner: https://api.github.com/users/johankitelman

#!/bin/sh
# Reset Parallels Desktop's trial and generate a casual email address to register a new user
rm /private/var/root/Library/Preferences/com.parallels.desktop.plist /Library/Preferences/Parallels/licenses.xml
jot -w pdu%d@gmail.com -r 1
