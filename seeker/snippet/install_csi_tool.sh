#date: 2024-07-09T16:57:03Z
#url: https://api.github.com/gists/482f0ab3291143bc8ded9188bc9c2e73
#owner: https://api.github.com/users/Stop423

#!/bin/bash

# Install development tools
sudo apt update
sudo apt install gcc make linux-headers-$(uname -r) git-core
# Build and install the modified wireless driver
CSITOOL_KERNEL_TAG=csitool-$(uname -r | cut -d . -f 1-2)
git clone https://github.com/posoo/linux-80211n-csitool.git --branch ${CSITOOL_KERNEL_TAG} --depth 1
cd linux-80211n-csitool
make -C /lib/modules/$(uname -r)/build M=$(pwd)/drivers/net/wireless/iwlwifi modules
sudo make -C /lib/modules/$(uname -r)/build M=$(pwd)/drivers/net/wireless/iwlwifi INSTALL_MOD_DIR=updates \
    modules_install
sudo depmod
cd ..
# Install the modified firmware
git clone https://github.com/posoo/linux-80211n-csitool-supplementary.git
for file in /lib/firmware/iwlwifi-5000-*.ucode
do 
    sudo mv $file $file.orig
done
sudo cp linux-80211n-csitool-supplementary/firmware/iwlwifi-5000-2.ucode.sigcomm2010 /lib/firmware/
sudo ln -s iwlwifi-5000-2.ucode.sigcomm2010 /lib/firmware/iwlwifi-5000-2.ucode
# Build the userspace logging tools
make -C linux-80211n-csitool-supplementary/netlink
echo "All Done!"