#date: 2022-01-06T17:22:20Z
#url: https://api.github.com/gists/cb7d19cad30dbd8b60891148bac564bb
#owner: https://api.github.com/users/cosmaioan

#!/bin/sh
# Initial set up script for Alpine as a Docker host in a Hyper-V VM
# Based on https://wiki.alpinelinux.org/wiki/Docker and https://wiki.alpinelinux.org/wiki/Hyper-V_guest_services
# Tested on Alpine 3.10
# wget -O - https://bit.ly/alpine-docker | sh

# Include community repo and update/upgrade
sed -i '/alpine\/v3.10\/community/s/^#*//g' /etc/apk/repositories
apk update
apk upgrade

# Enable public key authentication in sshd
#sed -i '/PubkeyAuthentication/s/^#*//g' /etc/ssh/sshd_config
#service sshd restart

# Add Hyper-V tools and services
apk add hvtools
rc-service hv_fcopy_daemon start
rc-service hv_kvp_daemon start
rc-service hv_vss_daemon start
rc-update add hv_fcopy_daemon
rc-update add hv_kvp_daemon
rc-update add hv_vss_daemon

# Enable swap accounting in kernel boot options
sed -i '/GRUB_CMDLINE_LINUX_DEFAULT/s/"$/ swapaccount=1"/' /etc/default/grub    
grub-mkconfig > /boot/grub/grub.cfg

# Install Docker and start at boot
apk add docker
rc-update add docker boot
service docker start

# Install docker-compose as a container
wget -O /usr/local/bin/docker-compose https://github.com/docker/compose/releases/download/1.24.1/run.sh
chmod +x /usr/local/bin/docker-compose

# Create docker user, enable and set authorized SSH key
echo Creating docker user...
adduser -DG docker -g 'Docker' docker
passwd -u docker
mkdir /home/docker/.ssh
chmod 700 /home/docker/.ssh
echo ssh-rsa authorized_key > /home/docker/.ssh/authorized_keys
chown -R docker.docker /home/docker/.ssh

# Power off
sync
poweroff