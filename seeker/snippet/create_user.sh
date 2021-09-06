#date: 2021-09-06T17:12:27Z
#url: https://api.github.com/gists/67849c356bf92127e28b387a7cf5630e
#owner: https://api.github.com/users/eggrollstan

#!/bin/sh

# https://apple.stackexchange.com/questions/82472/what-steps-are-needed-to-create-a-new-user-from-the-command-line/84039#84039

. /etc/rc.common
dscl . create /Users/administrator
dscl . create /Users/administrator RealName "Terminal User Account"
dscl . create /Users/administrator hint "Password Hint"
dscl . create /Users/administrator picture "/Path/To/Picture.png"
dscl . passwd /Users/administrator thisistheaccountpassword
dscl . create /Users/administrator UniqueID 550
dscl . create /Users/administrator PrimaryGroupID 20
dscl . create /Users/administrator UserShell /bin/bash
dscl . create /Users/administrator NFSHomeDirectory /Users/administrator
cp -R /System/Library/User\ Template/English.lproj /Users/administrator
chown -R administrator:staff /Users/administrator
