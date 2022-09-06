#date: 2022-09-06T17:11:47Z
#url: https://api.github.com/gists/6702bd3ae60aaf129318d5f86dd9a0cf
#owner: https://api.github.com/users/mariogarcia-ar

#source: https://computingforgeeks.com/install-virtualmin-on-centos-rhel-linux/ 

# Step 1: Update CentOS / RHEL system
sudo dnf update -y
sudo hostnamectl set-hostname <your-hostname>

# Step 2: Download Virtualmin install script
sudo dnf -y install wget
wget http://software.virtualmin.com/gpl/scripts/install.sh

#Step 3: Make the script executable and install Virtualmin
chmod a+x install.sh
sudo ./install.sh

#Step 4: Configure Firewall for Virtualmin on CentOS | RHEL 8
sudo firewall-cmd --zone=public --add-port=10000/tcp --permanent
sudo firewall-cmd --reload

#Step 5: Configure Virtualmin on CentOS | RHEL 8
https://<hostname>:10000



