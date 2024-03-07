#date: 2024-03-07T16:53:03Z
#url: https://api.github.com/gists/594e73607c4fc48f2fd16ad910177464
#owner: https://api.github.com/users/HNJAMeindersma

#!/bin/bash
# This script installs openHAB on a Debian Linux system anno 2024

# Update or fix all packages
sudo apt update &&
sudo apt upgrade -y &&
sudo apt install -y --fix-broken --fix-missing &&
sudo apt autoremove -y &&

# Set timezone
sudo timedatectl set-timezone Europe/Amsterdam &&

# Create required directories
sudo mkdir -p -v /usr/share/keyrings &&
sudo mkdir -p -v /etc/apt/sources.list.d &&

# Install additional basic packages
sudo apt install -y curl git wget zip unzip htop apt-transport-https gnupg ca-certificates qemu-guest-agent &&

# Install Java (Azul Zulu OpenJDK 17)
curl -s https://repos.azul.com/azul-repo.key | sudo gpg --dearmor -o /usr/share/keyrings/azul.gpg &&
echo "deb [signed-by=/usr/share/keyrings/azul.gpg] https://repos.azul.com/zulu/deb stable main" | sudo tee /etc/apt/sources.list.d/zulu.list &&
sudo apt update &&
sudo apt install -y zulu17-jdk &&

# Install openHAB
curl -s "https://openhab.jfrog.io/artifactory/api/gpg/key/public" | sudo gpg --dearmor -o /usr/share/keyrings/openhab.gpg &&
echo "deb [signed-by=/usr/share/keyrings/openhab.gpg] https://openhab.jfrog.io/artifactory/openhab-linuxpkg stable main" | sudo tee /etc/apt/sources.list.d/openhab.list &&
sudo apt update &&
sudo apt install -y openhab &&
sudo systemctl daemon-reload &&
sudo systemctl enable openhab.service &&
sudo systemctl start openhab.service &&

# Install arping (org.openhab.binding.network)
sudo apt install -y arping &&

# Install Ookla Speedtest CLI (org.openhab.binding.speedtest)
curl -s https://packagecloud.io/install/repositories/ookla/speedtest-cli/script.deb.sh | sudo bash &&
sudo apt update &&
sudo apt install -y speedtest &&

# Setup Samba shares
sudo apt install -y samba samba-common-bin &&
echo | sudo tee -a /etc/samba/smb.conf &&
echo "#======================= Custom Configuration =======================" | sudo tee -a /etc/samba/smb.conf &&
echo "wins support = yes" | sudo tee -a /etc/samba/smb.conf &&
echo "
[openHAB-userdata]
   comment = openHAB userdata
   path = /var/lib/openhab
   browseable = Yes
   writeable = Yes
   only guest = no
   public = no
   create mask = 0777
   directory mask = 0777" | sudo tee -a /etc/samba/smb.conf &&
echo "
[openHAB-conf]
   comment = openHAB site configuration
   path = /etc/openhab
   browseable = Yes
   writeable = Yes
   only guest = no
   public = no
   create mask = 0777
   directory mask = 0777" | sudo tee -a /etc/samba/smb.conf &&
echo "
[openHAB-logs]
   comment = openHAB logs
   path = /var/log/openhab
   browseable = Yes
   writeable = Yes
   only guest = no
   public = no
   create mask = 0777
   directory mask = 0777" | sudo tee -a /etc/samba/smb.conf &&
sudo sed -i '/\[homes\]/a \ \ \ available = no' /etc/samba/smb.conf &&
sudo sed -i '/\[print\$\]/a \ \ \ available = no' /etc/samba/smb.conf &&
(echo openhab; echo openhab) | sudo smbpasswd -a openhab -s &&
sudo systemctl restart smbd.service &&

# Output version information
sleep 5 && clear
echo "+---------------------------+"
echo "| OS information            |"
echo "+---------------------------+"
hostnamectl
echo
echo "+---------------------------+"
echo "| Timezone information      |"
echo "+---------------------------+"
timedatectl
echo
echo "+---------------------------+"
echo "| Java information          |"
echo "+---------------------------+"
java -version
echo
echo "+---------------------------+"
echo "| openHAB information       |"
echo "+---------------------------+"
openhab-cli info
echo
