#date: 2022-06-27T16:57:30Z
#url: https://api.github.com/gists/712b3c99b09c738ae9399adeae5365c3
#owner: https://api.github.com/users/mrobergl

# Find the appropriate version here http://download.virtualbox.org/virtualbox/
sudo VBOX_VERSION=6.0.4

sudo rpm -Uvh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
sudo rpm -Uvh http://download-ib01.fedoraproject.org/pub/epel/7/x86_64/Packages/d/dkms-2.6.1-1.el7.noarch.rpm
sudo yum -y install wget perl gcc dkms kernel-devel kernel-headers make bzip2

wget http://download.virtualbox.org/virtualbox/${VBOX_VERSION}/VBoxGuestAdditions_${VBOX_VERSION}.iso

sudo mkdir /media/VBoxGuestAdditions
sudo mount -o loop,ro VBoxGuestAdditions_${VBOX_VERSION}.iso /media/VBoxGuestAdditions

sudo sh /media/VBoxGuestAdditions/VBoxLinuxAdditions.run

rm -f VBoxGuestAdditions_${VBOX_VERSION}.iso
sudo umount /media/VBoxGuestAdditions
sudo rmdir /media/VBoxGuestAdditions
sudo unset VBOX_VERSION