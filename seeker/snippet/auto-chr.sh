#date: 2022-01-04T16:58:46Z
#url: https://api.github.com/gists/2b68a754d64b76391245ae793145e943
#owner: https://api.github.com/users/heirro

#!/usr/bin/bash
clear && \
echo "|=======================================================" && \
echo "|   Auto Installer Debian/Ubuntu" && \
echo "|   Created by Heirro Networking" && \
echo "|   Cloud Host Router - Long Term - Latest Version" && \
echo "|=======================================================" && \
while true
do
 read -r -p "Are you sure continue this Action? [Y/n] or [Yes/no] : " input
 
 case $input in
     [yY][eE][sS]|[yY])
clear
echo "|=======================================================" && \
echo "|   Auto Installer Debian/Ubuntu" && \
echo "|   Created by Heirro Networking" && \
echo "|   Cloud Host Router - Long Term - Latest Version" && \
echo "|=======================================================" && \
echo "| Installing Core Package" && \
apt-get -qq update && apt --fix-missing install -qq -y bzip2 gzip coreutils wget && apt-get install -qq build-essential -y && sysctl -w net.ipv6.conf.all.disable_ipv6=1 && sysctl -w net.ipv6.conf.default.disable_ipv6=1 && sysctl -w net.ipv6.conf.lo.disable_ipv6=1 && \
IPSAYA=$(curl http://ifconfig.io) && \
echo "| >> Installing Core Package [Success]" && \
sleep 2 && \
clear
echo "|=======================================================" && \
echo "|   Auto Installer Debian/Ubuntu" && \
echo "|   Created by Heirro Networking" && \
echo "|   Cloud Host Router - Long Term - Latest Version" && \
echo "|=======================================================" && \
echo "| Find Your Boot Disk Name (Ex: sda, sdb, vda, xvda, etc)" && \
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT && \
echo " ========================================================" && \
echo -n -e "| Choose Boot Disk Name : "; read diskme && \
clear
echo "|=======================================================" && \
echo "|   Auto Installer Debian/Ubuntu" && \
echo "|   Created by Heirro Networking" && \
echo "|   Cloud Host Router - Long Term - Latest Version" && \
echo "|=======================================================" && \
echo "| Find Your Network Interface Name (Ex: eth0, eth1, ens3, etc, not for lo)" && \
ip link && \
echo "| ========================================================" && \
echo -n -e "| Choose Network Interface Name: "; read netifme && \
clear
echo "|=======================================================" && \
echo "|   Auto Installer Debian/Ubuntu" && \
echo "|   Created by Heirro Networking" && \
echo "|   Cloud Host Router - Long Term - Latest Version" && \
echo "|=======================================================" && \
echo "| The installation process takes about 1-5 minutes."
sleep 1 && \
echo "| 1. Download RouterOS (CHR) Package" && \
wget -qq https://download.mikrotik.com/routeros/6.47.10/chr-6.47.10.img.zip -O chr.img.zip && \
sleep 1 && \
echo "| >> Download RouterOS (CHR) Package [Success]" && \
sleep 2 && \
echo "| 2. Unzip RouterOS (CHR) Package" && \
gunzip -c chr.img.zip > chr.img && \
sleep 1 && \
echo "| >> Unzip RouterOS (CHR) Package [Success]" && \
sleep 2 && \
echo "| 3. Mounting RouterOS (CHR) to Disk" && \
mount -o loop,offset=512 chr.img /mnt && \
ADDRESS=`ip addr show $netifme | grep global | cut -d' ' -f 6 | head -n 1` && \
GATEWAY=`ip route list | grep default | cut -d' ' -f 3` && \
echo "/ip address add address=$ADDRESS interface=[/interface ethernet find where name=ether1]
/ip route add gateway=$GATEWAY
" > /mnt/rw/autorun.scr && \
umount /mnt && \
echo u > /proc/sysrq-trigger && \
dd if=chr.img bs=1024 of=/dev/$diskme && \
sleep 1 && \
echo "| >> Mounting RouterOS (CHR) to Disk [Success]" && \
sleep 2 && \
echo "| All Installation success" && \
sleep 3 && \
echo "| Getting results" && \
sleep 4 && \
clear
echo "|=======================================================" && \
echo "|   Auto Installer Debian/Ubuntu" && \
echo "|   Created by Heirro Networking" && \
echo "|   Cloud Host Router - Long Term - Latest Version" && \
echo "|=======================================================" && \
echo "| Log in to the Winbox application "
echo "| Connect to: "$IPSAYA
echo "| Login: admin"
echo "| Password: (empty password, this default service)"
echo "| YOU MUST REBOOT ON VPS PANEL"
echo "| ======================================================="
 break
 ;;
     [nN][oO]|[nN])
clear
 echo "Cancelled by User"
 echo "If you want run again, just command -> ./install-chr.sh"
 break
        ;;
     *)
clear
echo "|=======================================================" && \
echo "|   Auto Installer Debian/Ubuntu" && \
echo "|   Created by Heirro Networking" && \
echo "|   Cloud Host Router - Long Term - Latest Version" && \
echo "|=======================================================" && \
 echo "Invalid input..."
 ;;
 esac
done