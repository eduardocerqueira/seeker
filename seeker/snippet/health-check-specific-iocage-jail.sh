#date: 2022-10-04T17:14:36Z
#url: https://api.github.com/gists/2e4137cddc4db0baec0b40bd6ea67d80
#owner: https://api.github.com/users/Erdack54

#!/bin/bash
#
ScriptName='health-check-specific-iocage-jail.sh'
ScriptVer='v1.5'
## by Erdack54 .fr /
## Usage&Crontab: sh /mnt/vdev0/dataset0/health-check-specific-iocage-jail.sh > /mnt/vdev0/dataset0/health-check-specific-iocage-jail.sh.log 2>&1
#



# Settings
IOCAGE_Jail_Adresse='172.16.0.2:8080'
IOCAGE_Jail_Name='qbittorrent'

# Colors /UNUSED fn
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' #NoColor

# Dependency Function: Check if script already try to restart a jails, if true exit
check_script_status(){
	if [ -f ${0%/*}/${0##*/}.last-try-to-restart.* ]
		then
		[ -f ${0%/*}/${0##*/}.last-try-to-restart.* ] && lastTryToRestart="$(basename -- ${0%/*}/${0##*/}.last-try-to-restart.*)" && echo  && echo "• Last (unsuccessful) try to Restart: ${lastTryToRestart##*.}"
		echo "|-- Doesn't respond multiples times and automatic restart doesn't seem to work, need human interaction, end of script as a preventive measure."
		echo "--------------------------------"
		exit 1
	else
		:
	fi
}

# Dependency Function: Check IOCAGE Jail Status
check_iocage_jail_status(){
	if iocage get -s $IOCAGE_Jail_Name 2>&1 | grep -w "up" ; then
		:
	else
		echo "down"
		echo "--------------------------------"
		exit 1
	fi
}

# Dependency Function: Check cURL
check_curl(){
	if curl --version 2>&1 | grep -w "libcurl" ; then
		:
	else
		echo "cURL is not installed :( Please install cURL before using this script. (Ex: 'apt-get install curl -y' for Debian/based OS's)"
		echo "--------------------------------"
		exit 1
	fi
}

# Primary Function: Health Check
health_check(){
	echo "--------------------------------"
	echo "$ScriptName $ScriptVer by Erdack54 .fr /"
	date +"%dd/%mm/%Yy %Hh:%Mm:%Ss"
	echo 
	echo "IOCAGE Jail:"
	echo "|-- Name: $IOCAGE_Jail_Name"
	echo "'-- Adresse+Port: $IOCAGE_Jail_Adresse"
	check_script_status
	[ -f ${0%/*}/${0##*/}.last-restart.* ] && lastRestart="$(basename -- ${0%/*}/${0##*/}.last-restart.*)" && echo  && echo "• Last Restart: ${lastRestart##*.}"
	echo 
	echo "----"
	echo 
	echo -e "• Jail Status:"
	check_iocage_jail_status
	echo 
	echo -e "• cURL Status:"
	check_curl
	echo 
	echo -e "• '$IOCAGE_Jail_Name' Status:"
	if timeout 3 curl -I "$IOCAGE_Jail_Adresse" 2>&1 | grep -w "200\|301" ; then
		:
	else
		health_re-check
	fi
	echo "--------------------------------"
}

# Secondary Function: Health Re-Check
health_re-check(){
	echo 
	echo -e "• Seem to be down, need to re-check ..."
	echo
	echo -e "• '$IOCAGE_Jail_Name' Status:"
	if timeout 5 curl -I "$IOCAGE_Jail_Adresse" 2>&1 | grep -w "200\|301" ; then
		:
	else
		echo 
		echo -e "• '$IOCAGE_Jail_Adresse' | '$IOCAGE_Jail_Name': is still down :( Try to automatically restarting it ..."
		if iocage restart $IOCAGE_Jail_Name 2>&1 | grep -w "Started OK" ; then
			rm -rf $lastRestart
			touch ${0%/*}/${0##*/}.last-restart.$(date +%dd-%mm-%Yy_%Hh%Mm%Ss)
			sleep 30
		else
			iocage stop $IOCAGE_Jail_Name
			rm -rf $lastRestart
			touch ${0%/*}/${0##*/}.last-try-to-restart.$(date +%dd-%mm-%Yy_%Hh%Mm%Ss)
			echo 
			echo -e "• $IOCAGE_Jail_Adresse | '$IOCAGE_Jail_Name': Doesn't respond multiples times and automatic restart doesn't seem to work, preventive measure actioned: this script will not re-run without human interaction."
		fi
	fi
}

# Execute: Health Check and Depandency aka the script :)
health_check

# End of Script, exit
exit
