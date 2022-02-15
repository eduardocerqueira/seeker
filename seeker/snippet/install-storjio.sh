#date: 2022-02-15T16:58:13Z
#url: https://api.github.com/gists/43bb50ccf8c270e46634b3427b74179d
#owner: https://api.github.com/users/tronblock

# !WARNING!

# Script moved to https://blob.CLITools.tk/install-storjio(https://github.com/marrobHD/clitools/blob/master/install-storjio)
# This script is no longer updated on GitHub Gist. 

# !WARNING!



#!/bin/bash
# How to use: wget -qO- CLITools.tk/install-storjio>install-storjio && chmod +x install-storjio && ./install-storjio STORJTOKEN LIVEPATCHTOKEN(optional)
# You can't directly wget it and run it with bash: wget xy | bash !
# Don't run the script with sudo!
cd ~
res1=$(date +%s.%N)


STORJTOKEN=${1?Format: ./install-storjio STORJTOKEN LIVEPATCHTOKEN(optional)}
LPTOKEN=${2:-nul}

while [ "$1" != "" ]; do
    case $1 in
        -c | --cleanup )   echo cleanup; rm ~\identity_linux_amd64.zip; exit 1
    esac
    shift
done


echo
echo
echo Version:	0.0.1 - Stable
echo Date:		29.06.2020
echo Author:	TechHome/marrobhd
echo Source:	info.clitools.tk
echo
echo

sudo apt-get install curl

echo Setting up variables...
echo ----------------------------------------------------------
echo
function pause(){
read -s -n 1 -p "Press any key to continue ..."
echo
}

# Ask for the user password
# Script only works if sudo caches the password for a few minutes
sudo true

sudo rm /run/shm/error.log

IP=$(hostname -I | cut -d' ' -f1)
PUBLICIP=$(curl ifconfig.me)
USERNAME=$(whoami)
HOMEDIR=$(echo ~)

if [ $LPTOKEN == "nul" ] 2>/run/shm/error.log
  then
    echo "We will let livepatch uninstalled!"
  else
    echo "We will install and enable livepach!"
fi
echo storjtoken: $STORJTOKEN
echo lptoken: $LPTOKEN
echo Please verify your details
pause
echo ----------------------------------------------------------
echo Success!
echo

#intall storjo
echo
echo
echo Install required packages to build...
echo ----------------------------------------------------------

if sudo apt-get update && sudo apt-get install snapd unzip openssh-server -y; then
    echo
    echo
else
    echo Trying to fix errors...
    sudo dpkg --configure -a || exit 1
    sudo apt-get update && apt-get install sudo && sudo apt-get dist-upgrade -y && sudo apt-get install snapd unzip openssh-server -y || exit 1
fi

echo
# Install docker and docker compose via script 
wget -qO- https://gist.github.com/marrobHD/42f1ab222d4f83e811b15bae72a346ea/raw | bash

# optional: install and enable livepatch on your system
if [ "$LPTOKEN" != "nul" ]
then
	echo
	echo
	echo Installing and enabling canonical-livepatch on your system...
	echo ----------------------------------------------------------
	sudo snap install canonical-livepatch && sudo canonical-livepatch enable $LPTOKEN
	echo ----------------------------------------------------------
	echo
else
	echo
fi

echo
echo ----------------------------------------------------------
echo Success!



echo
echo

echo Creating/Importing an identity...
echo ----------------------------------------------------------
echo
mkdir -p ~/.local/share/storj/identity/
echo If you want to import an identity, copy your backed up folder "storagenode" to ~/.local/share/storj/identity/
echo If you did all well the script detects it and skips the creation of a new identity.
echo
echo 1. login to $IP with username $USERNAME and your password
echo 2. go to ~/.local/share/storj/identity/ and paste the directory "storagenode"
echo
echo If you want to create a new identity you can ignore this.
echo
pause
echo
curl -L https://github.com/storj/storj/releases/latest/download/identity_linux_amd64.zip -o identity_linux_amd64.zip || exit 1
unzip -o identity_linux_amd64.zip || exit 1
chmod +x identity || exit 1
sudo mv identity /usr/local/bin/identity || exit 1
echo

if grep -c BEGIN ~/.local/share/storj/identity/storagenode/ca.cert | grep -q '2'; then
        echo "already matched"
        if grep -c BEGIN ~/.local/share/storj/identity/storagenode/identity.cert | grep -q '3'; then
                echo "already matched"
				IDENTITYSTATUS=matched
        else
                echo "mismatched"
        fi
else
    echo "No existing identity found. Create one?"
	
fi

if [ "$IDENTITYSTATUS" != "matched" ]; then
while true; do
    read -p "Creating an identity can take several hours or days, proceed? " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) sudo rm identity_linux_amd64.zip && sudo rm -r /usr/local/bin/identity; exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
fi

if [ "$IDENTITYSTATUS" != "matched" ]; then
identity create storagenode || exit 1
identity authorize storagenode $STORJTOKEN
fi

grep -c BEGIN ~/.local/share/storj/identity/storagenode/ca.cert | grep -q '2' && echo matched || echo mismatched || exit 1
grep -c BEGIN ~/.local/share/storj/identity/storagenode/identity.cert | grep -q '3' && echo matched || echo mismatched || exit 1


echo
echo ----------------------------------------------------------
echo Sucess

echo
echo
echo Backup your identity!
echo ----------------------------------------------------------
echo
echo Backup before you continue, it should be quick!
echo This allows you to restore your Node in case of an unfortunate hardware or OS incident.
echo
echo 1. login to $IP with username $USERNAME and your password
echo 2. go to ~/.local/share/storj/identity/ and back up the directory "storagenode"
echo
while true; do
    read -p "I followed the steps and have now saved the storagenode folder. " yn
    case $yn in
        [Yy]* ) echo && echo Great, lets continue; break;;
        [Nn]* ) echo && echo Your own risk! I recommend do this later, lets continue; break;;
        * ) echo "Please answer yes or no.";;
    esac
done
 
echo
echo ----------------------------------------------------------
echo Finished!


echo
echo
echo CLI Install via Docker/Compose
echo ----------------------------------------------------------
echo
cd ~

echo "Enter external(port to forward) eg. 28967 and press [ENTER]: "
read STORJPORT1
echo
echo "Enter webdashport(port for web dashboard) eg. 14002 and press [ENTER]: "
read STORJPORT2
echo
echo "Enter myetherwallet.com Wallet ID eg. 0xe5380DBa756Ea63b2bc98769bEB614750DaE0089 and press [ENTER]: "
read WALLET
echo
echo "Enter your email address from your storjtoken and press [ENTER]: "
read EMAIL
echo
echo "Enter your full domain eg. storjio.mydomain.com:$STORJPORT1, which points to $PUBLICIP [ENTER]: "
read FULLDOMAIN
echo
echo "Size of Storage + 10% overhead. If you over-allocate space, you may corrupt your database! ex. 600GB disk, 500GB + 10%=550GB [ENTER]: "
read STORAGE
echo
echo "Enter the path where storjio stores its files. eg. /mnt/storjio or $HOMEDIR/storagenode_files [ENTER]: "
read STORAGEPATH
echo


echo
echo Do your port forwarding: From: Anywhere, Port $STORJPORT1, Forward IP: $IP, Forward Port: $STORJPORT1, Protocol: TCP
echo
pause


echo "
version: \"3.3\"
services:
  storagenode:
    image: storjlabs/storagenode:beta
    volumes:
      - type: bind
        source: \"$HOMEDIR/.local/share/storj/identity/storagenode\"
        target: /app/identity
      - type: bind
        source: \"$STORAGEPATH\"
        target: /app/config
    ports:
      - $STORJPORT1:28967
      - $STORJPORT2:14002
    deploy:
      restart_policy:
        condition: always
    environment:
      - WALLET=$WALLET
      - EMAIL=$EMAIL
      - ADDRESS=$FULLDOMAIN
      - STORAGE=$STORAGE

" | tee docker-compose.yml > /dev/null

echo "
# /etc/systemd/system/storagenode.service

[Unit]
Description=Storagenode
Requires=docker.service
After=docker.service


[Service]
Type=simple
RemainAfterExit=yes
WorkingDirectory=$HOMEDIR
ExecStart=/usr/local/bin/docker-compose up
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0
#Restart=always

[Install]
WantedBy=multi-user.target

" | sudo tee /etc/systemd/system/storagenode.service > /dev/null

mkdir ~/storagenode_files
sudo usermod -aG docker $USER

echo
sudo systemctl daemon-reload

sudo systemctl restart storagenode.service
sleep 6
sudo systemctl status storagenode.service
sudo systemctl enable storagenode.service

echo "watch systemctl status storagenode.service">watch.sh && chmod +x watch.sh
echo "sudo docker exec -it storjio_storagenode_1 /app/dashboard.sh">dashboard.sh && chmod +x dashboard.sh

sudo docker run -d --restart=always --name watchtower -v /var/run/docker.sock:/var/run/docker.sock storjlabs/watchtower storagenode watchtower --stop-timeout 300s --interval 21600
sudo docker ps -a

echo
echo ----------------------------------------------------------
echo Finished!
res2=$(date +%s.%N)
dt=$(echo "$res2 - $res1" | bc)
dd=$(echo "$dt/86400" | bc)
dt2=$(echo "$dt-86400*$dd" | bc)
dh=$(echo "$dt2/3600" | bc)
dt3=$(echo "$dt2-3600*$dh" | bc)
dm=$(echo "$dt3/60" | bc)
ds=$(echo "$dt3-60*$dm" | bc)
LC_NUMERIC=C printf "Total runtime: %d:%02d:%02d:%02.4f\n" $dd $dh $dm $ds
echo
echo
echo Now point your browser to $IP:$STORJPORT2 and you will see the beautiful STORJ dashboard. If not check the docker_compose.yml and find issues with watch systemctl status storagenode.service.
echo
echo Tip: Run this script again with var cleanup to delete all files that were needed during installation progress.
exit

