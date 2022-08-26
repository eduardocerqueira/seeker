#date: 2022-08-26T16:49:57Z
#url: https://api.github.com/gists/6b28a706ed30ab8ce7913f00b9ad94a0
#owner: https://api.github.com/users/Fire7ly

#!/bin/bash

# Author : FIRE7LY
# Telegram : @fire7ly
# A script to enable ssh after first configuration of Ubport ...
# This script made in hope that it will usefull to automate the ssh configuration processes
# To avoide all the tings doing menually it will saye time and increase speed of debigging tihings...
# All the conmmand and instructions are taken from official ubport documentation (https://docs.ubports.com/en/latest/userguide/advanceduse/ssh.html)  ... 
# So all credit goies to them..
###################################################################################################################################################

#Env Ver
home="/home/phablet"
key="id_rsa.pub"
auth_key="authorized_keys"
log="$home/ssh.log"

# Banner
welcome () {

    echo "====================================================="
    echo "|                       UBPORT                      |"
    echo "|                     SSH ENBALER                   |"
    echo "|               MADE WITH ❤️ BY FIRE7LY              |"
    echo "====================================================="

}

# Check If User Push ssh Key Into Home Directory ...
is_key_present () {
    if [ -f $home/$key ]; then
        return 0
    else
        return 1
    fi
}


#Check SSH Folder
is_ssh_dir_present () {
    if [ -d $home/.ssh ]; then 
        echo "SSH: Key Present .."
    else
        mkdir $home/.ssh
    fi

    # Change Permisssion of folder..
    chmod 700 $home/.ssh
}



#Setup SSH Keys 
set_key () {

    cat  $home/$key >> $home/.ssh/$auth_key
    [ -f $home/.ssh/$auth_key ] && chmod 600 $home/.ssh/$auth_key && chown -R phablet:phablet $home/.ssh/$auth_key

}

#Start SSH Services
start_ssh () {

    if sudo android-gadget-service enable ssh; then
        echo "SSH: ssh Service Started ..."
    else
        exit 1
    fi

}


#Check All Things
check () {
    
    if is_key_present; then 
        is_ssh_dir_present
    else
        echo "Please Put SSH Keys In $home"
        exit 0
    fi
}


main () {

    ip=$(hostname -I | awk '{print $2}')
    welcome
    check
    set_key
    echo "SSH: Key Setup Successfully ..."
    if start_ssh; then 
        echo "SSH: SSH IS RUNNING YOU CAN CONNECT NOW : $(whoami)@$ip"
    else 
        echo "SSH: Somthing Wrong !!! \nCheck Logs And Run Me Again ..."
    fi
}


packet=$(ping -q -c 1 -W 1 8.8.8.8 | grep "packet loss" | awk '{ print $6 }')

if [[ $packet = "0%" ]]; then
    echo "SSH: Internet Is Up ..."
    main | tee $log
else
    welcome | tee $log
    echo "SSH: Internet Is Down ... \nPlease Check Your Wifi" | tee $log
    exit 1
fi