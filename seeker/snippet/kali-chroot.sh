#date: 2022-02-21T16:52:50Z
#url: https://api.github.com/gists/3b0b0bd37c8b76bc2bc7fc10530c1f1d
#owner: https://api.github.com/users/kamakazix

#!/bin/bash

if (( $EUID != 0 )); then
    echo "Please run as root"
    exit
fi

if ! debootstrap --help 2>/dev/null | grep -q 'Debian'; then
   echo "Please install debootstrap"
   exit 1
fi

if ! ping 1.1.1.1 2>/dev/null | grep -q "ttl="; then
   echo "Please check your Internet"
   exit
fi

distro=kali-rolling
mirror=http://kali.download/kali

rootfsDir=/kali-chroot
Binary=/usr/local/bin/kali-chroot

if [ -d "${rootfsDir}" ]; then
    echo "${rootfsDir} already exists"
    exit
fi 

rootfs_chroot() {
        PATH='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' \
                chroot "$rootfsDir" "$@"
}

debootstrap --components=main,contrib,non-free \
    --include=kali-archive-keyring \
    $distro $rootfsDir $mirror

rootfs_chroot apt-get -y install nano curl wget git netcat python2 python3 python3-pip p7zip-full p7zip-rar p7zip unzip net-tools build-essential iputils-ping iproute2 pciutils bash-completion kali-linux-core
rootfs_chroot apt-get -y autoremove
rootfs_chroot apt-get -y clean
rootfs_chroot wget https://gist.githubusercontent.com/scarfaced18/aabefc19aeabc8d21771ad1a678e0b63/raw/942fb264b44806b6f6866016c201c0cb557c7666/.bashrc -O /root/.bashrc
rootfs_chroot wget https://gist.githubusercontent.com/scarfaced18/077331abf1671028aecbf392e79d8cd0/raw/82fa05d6797d42b525a18071c6d746289df9b51c/.bash_profile -O /root/.bash_profile

cat <<EOT >> $Binary
#!/bin/bash
if (( SOMEEUID != 0 )); then   
    echo "Please run as root"
    exit
fi
xhost +local:root &>/dev/null && chroot ${rootfsDir} && xhost -local:root &>/dev/null
EOT
sed -i 's/SOMEEUID/"$EUID"/g' $Binary
chmod +x $Binary