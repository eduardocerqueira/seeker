#date: 2023-05-15T16:59:55Z
#url: https://api.github.com/gists/816c74da802a91077f9ddbb9c12ff1e8
#owner: https://api.github.com/users/zoeeechu

#!/bin/bash

####################### IMPORTANT #######################
# REQUIRES 
# https://github.com/zoeeechu/MicrosoftFixer/blob/main/Microsoft_Office_License_Removal.pkg 
# to be in same DIR
#
# check out the repo for full explination <3
# zoeeechu/MicrosoftFixer
#########################################################

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

BASEDIR=$(dirname $0)

#rm bellow.
#Microsoft Office Identities cache 2
#Microsoft Office Identities settings 2 
##

security delete-generic-password -l com.microsoft.adalcache


cd ~/Library
cd "Group Containers"

rm -r  UBF8T346G9.ms
rm -r  UBF8T346G9.Office
rm -r  UBF8T346G9.OfficeOsfWebHost



cd ${BASEDIR}
sudo installer -package "Microsoft_Office_License_Removal.pkg" -target /



pGreen "done."

pause