#date: 2023-05-15T17:01:39Z
#url: https://api.github.com/gists/9a72e455eb3bac8844bfa21fea3a50f5
#owner: https://api.github.com/users/zoeeechu

#!/bin/bash

function pause(){
 read -s -n 1 -p "Press any key to continue . . ."
 echo ""
}


#print in yellow
pYellow() {
   printf "\e[33;1m%s\n" "$1"
}

#print in green
pGreen() {
   printf "\e[32m$1\e[m\n"
}

#print in red
pRed() {
   echo -e "\033[1;31m$1\033[0m"
}

#reset colors
r() {
   tput init
}

#if not root throw error
if [ $USER != "root" ]; then
sudo "$0" "$@"
exit $?
fi

#Get DIR
BASEDIR=$(dirname $0)


cd ${BASEDIR}

for i in *.cer; do
sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain "$i"
done

pGreen "done."

pause