#date: 2024-01-11T16:47:12Z
#url: https://api.github.com/gists/8fca41b2ab763fbe69e653159ed6a5c9
#owner: https://api.github.com/users/fbottarel

#!/usr/bin/bash

# Apply the PREEMPT_RT patch to linux kernel 6.1

# Get source code for the kernel and patch
# Match kernel and patch versions using 
# https://cdn.kernel.org/pub/linux/kernel/v6.x/
# and
# https://wiki.linuxfoundation.org/realtime/preempt_rt_versions
mkdir -p ~/rt_kernel_6.1.70
cd ~/rt_kernel_6.1.70
wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.1.70.tar.gz
wget https://cdn.kernel.org/pub/linux/kernel/projects/rt/6.1/patch-6.1.70-rt21.patch.xz

# Unpack and apply the patch
tar -xzf linux-6.1.70.tar.gz
xz -d patch-6.1.70-rt21.patch.xz
cd linux-6.1.70
patch -p1 <../patch-6.1.70-rt21.patch

# Install build dependencies
sudo apt update
sudo apt install make gcc libncurses-dev libssl-dev flex libelf-dev bison

# Copy whatever kernel config file is already present on the machine, possibly close to the patched one
cp /boot/config-6.2.0-39-generic .config

# Configure the build process for the RT preemtpion model
make menuconfig

# General setup -> Preemption Model -> Fully Preemptible Kernel (Real Time)
# Select -> Save -> Exit
# Add any additional configuration 

# Compile and install
sudo make -j4
sudo make modules_install
sudo make install

# Reboot and select the "Linux 6.1.70-rt21" kernel from the GRUB choice
sudo reboot

# Check if the new kernel is actually in use
uname -a 

# If the GRUB screen does not show up at reboot...

# If you get a 
# make[1]: *** No rule to make target 'debian/canonical-certs.pem', needed by 'certs/x509_certificate_list'.  Stop.
# error at compile time, you can disable kernel signing. Before compiling (after menuconfig) run 
cd ~/rt_kernel_6.1.70
scripts/config --disable SYSTEM_TRUSTED_KEYS
scripts/config --disable SYSTEM_REVOCATION_KEYS
# If you absolutely need to sign it (e.g. secure boot) you can generate a signature certificate and whitelist it.
# Follow the guidelines here 
# https://askubuntu.com/a/1372567
# I did not sign it as I am lazy and it pretty late.


