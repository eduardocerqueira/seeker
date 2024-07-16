#date: 2024-07-16T17:07:36Z
#url: https://api.github.com/gists/dc7c300f12611f7c116dcf31f5c6c2af
#owner: https://api.github.com/users/RCold

#!/bin/sh

#DEBIAN_SUITE='bookworm'
#DEBIAN_KERNEL_FLAVOR='generic'
#MIRROR_HOSTNAME='deb.debian.org'
MIRROR_DIRECTORY='/debian'
#SECURITY_HOSTNAME='security.debian.org'
#DEBIAN_HOSTNAME='debian'
#INTERFACE='eth0'
#IPADDRESS='192.168.1.100'
#NETMASK='255.255.255.0'
#GATEWAY='192.168.1.1'
#NAMESERVERS='8.8.8.8 8.8.4.4'
#DHCP='yes'
#ROOT_PASSWORD= "**********"
#USERNAME='rcold'
#FULLNAME='Yeuham Wang'
#USER_PASSWORD= "**********"
#SSH_KEY=''
#SSH_PORT='22'
#SSH_PERMIT_ROOT_LOGIN='no'
#SSH_PASSWORD_AUTHENTICATION= "**********"
#SYS_DISK='/dev/sda'
SWAP_SIZE='512'
PACKAGES='aptitude ca-certificates dbus pciutils systemd-timesyncd wget zstd'
#ENABLE_SYSTEMD_NETWORKD='yes'
TIMEZONE='Asia/Shanghai'
CMDLINE_LINUX='net.ifnames=0 biosdevname=0'

set -e

netmask_prefix() {
  mask=$(echo "$1" | sed 's/\./\\./g')
  cat << EOF | grep -E "^$mask/|/$mask\$"
128.0.0.0/1
192.0.0.0/2
224.0.0.0/3
240.0.0.0/4
248.0.0.0/5
252.0.0.0/6
254.0.0.0/7
255.0.0.0/8
255.128.0.0/9
255.192.0.0/10
255.224.0.0/11
255.240.0.0/12
255.248.0.0/13
255.252.0.0/14
255.254.0.0/15
255.255.0.0/16
255.255.128.0/17
255.255.192.0/18
255.255.224.0/19
255.255.240.0/20
255.255.248.0/21
255.255.252.0/22
255.255.254.0/23
255.255.255.0/24
255.255.255.128/25
255.255.255.192/26
255.255.255.224/27
255.255.255.240/28
255.255.255.248/29
255.255.255.252/30
255.255.255.254/31
255.255.255.255/32
EOF
}

if [ "$(id -u)" -ne 0 ]; then
  echo 'This script must be run as root!' >&2
  exit 1
fi

if [ "$(uname -m)" = 'x86_64' ]; then
  DEBIAN_ARCH='amd64'
else
  echo 'Your machine hardware is not supported.' >&2
  exit 1
fi

if [ -r /etc/os-release ]; then
  . /etc/os-release
else
  echo 'Your Linux distribution is not supported.' >&2
  exit 1
fi

if [ "$ID" != 'alpine' ]; then
  echo 'Your Linux distribution is not supported.' >&2
  exit 1
fi

if [ -z "$DEBIAN_SUITE" ]; then
  echo 'Which version of Debian do you want to use?'
  echo '  1: Debian 12 (bookworm)'
  echo '  2: Debian 11 (bullseye)'
fi

until [ -n "$DEBIAN_SUITE" ]; do
  printf 'Enter your choice: [1] '
  read -r DEBIAN_SUITE_CHOICE
  case $DEBIAN_SUITE_CHOICE in
    1|'')
      DEBIAN_SUITE='bookworm'
      ;;
    2)
      DEBIAN_SUITE='bullseye'
      ;;
  esac
done

if [ "$DEBIAN_KERNEL_FLAVOR" != 'generic' ] && [ "$DEBIAN_KERNEL_FLAVOR" != 'cloud' ]; then
  echo "Which flavor of Debian kernel do you want to use?"
  echo '  1: generic'
  echo '  2: cloud'
fi

until [ "$DEBIAN_KERNEL_FLAVOR" = 'generic' ] || [ "$DEBIAN_KERNEL_FLAVOR" = 'cloud' ]; do
  printf 'Enter your choice: [1] '
  read -r DEBIAN_KERNEL_FLAVOR_CHOICE
  case $DEBIAN_KERNEL_FLAVOR_CHOICE in
    1|'')
      DEBIAN_KERNEL_FLAVOR='generic'
      ;;
    2)
      DEBIAN_KERNEL_FLAVOR='cloud'
      ;;
  esac
done

DEFAULT_MIRROR_HOSTNAME='deb.debian.org'
if ${MIRROR_HOSTNAME+false}; then
  printf 'Enter mirror hostname: [%s] ' "$DEFAULT_MIRROR_HOSTNAME"
  read -r MIRROR_HOSTNAME
  if [ -z "$MIRROR_HOSTNAME" ]; then
    MIRROR_HOSTNAME=$DEFAULT_MIRROR_HOSTNAME
  fi
fi

DEFAULT_SECURITY_HOSTNAME='security.debian.org'
if ${SECURITY_HOSTNAME+false}; then
  printf 'Enter security mirror hostname: [%s] ' "$DEFAULT_SECURITY_HOSTNAME"
  read -r SECURITY_HOSTNAME
fi
if [ -z "$SECURITY_HOSTNAME" ]; then
  SECURITY_HOSTNAME=$DEFAULT_SECURITY_HOSTNAME
fi

DEFAULT_HOSTNAME=$(hostname)
until [ -n "$DEBIAN_HOSTNAME" ]; do
  if ${DEBIAN_HOSTNAME+false} || [ -z "$DEFAULT_HOSTNAME" ]; then
    printf 'Enter system hostname: '
    if [ -n "$DEFAULT_HOSTNAME" ]; then
     printf '[%s] ' "$DEFAULT_HOSTNAME"
    fi
    read -r DEBIAN_HOSTNAME
  fi
  if [ -z "$DEBIAN_HOSTNAME" ]; then
    DEBIAN_HOSTNAME=$DEFAULT_HOSTNAME
  fi
done

DEFAULT_INTERFACE=$(ip -4 route list default 2>/dev/null | grep -Eom1 ' dev [^ ]+' | awk '{print $2}')
if [ "$DHCP" != 'yes' ] && [ -n "$DEFAULT_INTERFACE" ]; then
  IPSUB=$(ip -4 address show dev "$DEFAULT_INTERFACE" scope global 2>/dev/null | grep -Eom1 '^ *inet [^ ]+' | awk '{print $2}')
  if [ -n "$IPSUB" ]; then
    DEFAULT_IPADDRESS=$(echo "$IPSUB" | cut -d'/' -f1)
    DEFAULT_NETMASK=$(echo "$IPSUB" | cut -d'/' -f2)
    if [ -z "$DEFAULT_NETMASK" ]; then
      DEFAULT_NETMASK='32'
    fi
  fi
  DEFAULT_GATEWAY=$(ip -4 route list default 2>/dev/null | grep -Eom1 ' via [^ ]+' | awk '{print $2}')
  DEFAULT_NAMESERVERS=$(awk '$1 == "nameserver" && $2 !~ /^127\./ {print $2}' /etc/resolv.conf | sed ':a;N;s/\n/ /;ta')
fi

until [ -n "$INTERFACE" ]; do
  if ${INTERFACE+false} || [ -z "$DEFAULT_INTERFACE" ]; then
    printf "Which network interface do you want to use? "
    if [ -n "$DEFAULT_INTERFACE" ]; then
      printf '[%s] ' "$DEFAULT_INTERFACE"
    fi
    read -r INTERFACE
  fi
  if [ -z "$INTERFACE" ]; then
    INTERFACE=$DEFAULT_INTERFACE
  fi
done

until [ "$DHCP" = 'yes' ] || [ -n "$IPADDRESS" ]; do
  if ${IPADDRESS+false} || [ -z "$DEFAULT_IPADDRESS" ]; then
    printf "IP address for %s? (or 'dhcp') " "$INTERFACE"
    if [ -n "$DEFAULT_IPADDRESS" ]; then
     printf '[%s] ' "$DEFAULT_IPADDRESS"
    fi
    read -r IPADDRESS
  fi
  if [ "$IPADDRESS" = 'dhcp' ]; then
    DHCP='yes'
    unset IPADDRESS
  elif [ -z "$IPADDRESS" ]; then
    IPADDRESS=$DEFAULT_IPADDRESS
  fi
done

if [ -n "$NETMASK" ]; then
  NETMASK=$(netmask_prefix "$NETMASK" | cut -d'/' -f2)
  if [ -z "$NETMASK" ]; then
    unset NETMASK
  fi
fi
until [ "$DHCP" = 'yes' ] || [ -n "$NETMASK" ]; do
  if ${NETMASK+false} || [ -z "$DEFAULT_NETMASK" ]; then
    printf 'Netmask? '
    if [ -n "$DEFAULT_NETMASK" ]; then
      printf '[%s] ' "$(netmask_prefix "$DEFAULT_NETMASK" | cut -d'/' -f1)"
    fi
    read -r NETMASK
  fi
  if [ -z "$NETMASK" ]; then
    NETMASK=$DEFAULT_NETMASK
  else
    NETMASK=$(netmask_prefix "$NETMASK" | cut -d'/' -f2)
    if [ -z "$NETMASK" ]; then
      unset NETMASK
    fi
  fi
done

until [ "$DHCP" = 'yes' ] || [ -n "$GATEWAY" ]; do
  if ${GATEWAY+false} || [ -z "$DEFAULT_GATEWAY" ]; then
    printf 'Gateway? '
    if [ -n "$DEFAULT_GATEWAY" ]; then
      printf '[%s] ' "$DEFAULT_GATEWAY"
    fi
    read -r GATEWAY
  fi
  if [ -z "$GATEWAY" ]; then
    GATEWAY=$DEFAULT_GATEWAY
  fi
done

until [ "$DHCP" = 'yes' ] || [ -n "$NAMESERVERS" ]; do
  if ${NAMESERVERS+false} || [ -z "$DEFAULT_NAMESERVERS" ]; then
    printf 'DNS nameserver(s)? '
    if [ -n "$DEFAULT_NAMESERVERS" ]; then
      printf '[%s] ' "$DEFAULT_NAMESERVERS"
    fi
    read -r NAMESERVERS
  fi
  if [ -z "$NAMESERVERS" ]; then
    NAMESERVERS=$DEFAULT_NAMESERVERS
  fi
done

if ${ROOT_PASSWORD+false}; then
  printf "Enter password for root: "**********"
  read -r ROOT_PASSWORD
fi
if [ "$ROOT_PASSWORD" = "**********"
  ROOT_PASSWORD= "**********"
fi

if ${USERNAME+false}; then
  printf "Setup a user? (enter a lower-case loginname, or 'no') [no] "
  read -r USERNAME
  if [ -n "$USERNAME" ] && [ "$USERNAME" != 'no' ]; then
    if ${FULLNAME+false}; then
      printf 'Full name for user %s? [%s] ' "$USERNAME" "$USERNAME"
      read -r FULLNAME
      if [ -z "$FULLNAME" ]; then
        FULLNAME=$USERNAME
      fi
    fi
    if ${USER_PASSWORD+false}; then
      printf "Enter password for %s: "**********"
      read -r USER_PASSWORD
    fi
  fi
fi
if [ "$USERNAME" = 'no' ]; then
  USERNAME=''
fi
if [ "$USER_PASSWORD" = "**********"
  USER_PASSWORD= "**********"
fi

if [ -n "$USERNAME" ]; then
  if [ "$SSH_PASSWORD_AUTHENTICATION" != "**********"= 'no' ]; then
    unset SSH_PASSWORD_AUTHENTICATION
  fi
  while ${SSH_PASSWORD_AUTHENTICATION+false}; do
    printf 'Allow SSH login with password? (y/n) [y] '
    read -r SSH_PASSWORD_AUTHENTICATION
    if [ -z "$SSH_PASSWORD_AUTHENTICATION" ] || expr "$SSH_PASSWORD_AUTHENTICATION"  : "**********"
      SSH_PASSWORD_AUTHENTICATION= "**********"
    elif expr "$SSH_PASSWORD_AUTHENTICATION"  : "**********"
      SSH_PASSWORD_AUTHENTICATION= "**********"
    else
      unset SSH_PASSWORD_AUTHENTICATION
    fi
  done
  PACKAGES="sudo $PACKAGES"
else
  if [ "$SSH_PERMIT_ROOT_LOGIN" != "**********"= 'no' ] && [ "$SSH_PERMIT_ROOT_LOGIN" != 'prohibit-password' ]; then
    unset SSH_PERMIT_ROOT_LOGIN
  fi
  DEFAULT_SSH_PERMIT_ROOT_LOGIN= "**********"
  while ${SSH_PERMIT_ROOT_LOGIN+false}; do
    printf 'Allow root SSH login? (? for help) [%s] ' "$DEFAULT_SSH_PERMIT_ROOT_LOGIN"
    read -r SSH_PERMIT_ROOT_LOGIN
    if [ -z "$SSH_PERMIT_ROOT_LOGIN" ]; then
      SSH_PERMIT_ROOT_LOGIN=$DEFAULT_SSH_PERMIT_ROOT_LOGIN
    elif [ "$SSH_PERMIT_ROOT_LOGIN" = '?' ]; then
      echo 'Valid options are: '
      echo '  yes                root will be able to login with password or SSH key'
      echo '  no                 root will not be allowed to login with SSH'
      echo '  prohibit-password  root will be able to login with SSH key but not with'
      echo '                     password'
      unset SSH_PERMIT_ROOT_LOGIN
    elif [ "$SSH_PERMIT_ROOT_LOGIN" != "**********"= 'no' ] && [ "$SSH_PERMIT_ROOT_LOGIN" != 'prohibit-password' ]; then
      unset SSH_PERMIT_ROOT_LOGIN
    fi
  done
fi

if ${SSH_KEY+false}; then
  if [ -z "$USERNAME" ]; then
    printf "Enter SSH key for root: (or 'none') [none] "
  else
    printf "Enter SSH key for %s: (or 'none') [none] " "$USERNAME"
  fi
  read -r SSH_KEY
fi
if [ "$SSH_KEY" = 'none' ]; then
  SSH_KEY=''
fi

if ! [ -b "$SYS_DISK" ]; then
  if [ -n "$SYS_DISK" ]; then
    unset SYS_DISK
  fi
  if [ -b /dev/sda ]; then
    DEFAULT_SYS_DISK='/dev/sda'
  elif [ -b /dev/vda ]; then
    DEFAULT_SYS_DISK='/dev/vda'
  fi
fi

ERASE_DISK='y'
until [ -b "$SYS_DISK" ]; do
  if ${SYS_DISK+false} || ! [ -b "$DEFAULT_SYS_DISK" ]; then
    printf 'Which disk do you want to use? '
    if [ -b "$DEFAULT_SYS_DISK" ]; then
      printf '[%s] ' "$DEFAULT_SYS_DISK"
    fi
    read -r SYS_DISK
    unset ERASE_DISK
  fi
  if [ -z "$SYS_DISK" ]; then
    SYS_DISK=$DEFAULT_SYS_DISK
  elif ! [ -b "$SYS_DISK" ]; then
    unset SYS_DISK
  fi
done

until expr "$ERASE_DISK" : '[Yy]$' >/dev/null; do
  printf 'WARNING: Erase the disk and continue? (y/n) [n] '
  read -r ERASE_DISK
  if [ -z "$ERASE_DISK" ] || expr "$ERASE_DISK" : '[Nn]$' >/dev/null; then
    exit 1
  fi
done

if [ -e /dev/virtio-ports/org.qemu.guest_agent.0 ]; then
  PACKAGES="qemu-guest-agent $PACKAGES"
elif [ -e /dev/vmbus/hv_fcopy ] || [ -e /dev/vmbus/hv_kvp ] || [ -e /dev/vmbus/hv_vss ]; then
  PACKAGES="hyperv-daemons $PACKAGES"
fi
if [ "$DEBIAN_SUITE" = 'bookworm' ] && [ "$ENABLE_SYSTEMD_NETWORKD" = 'yes' ]; then
  PACKAGES="systemd-resolved $PACKAGES"
fi
case $DEBIAN_KERNEL_FLAVOR in
  generic)
    PACKAGES="grub-pc linux-image-$DEBIAN_ARCH openssh-server $PACKAGES"
    ;;
  cloud)
    PACKAGES="grub-pc linux-image-cloud-$DEBIAN_ARCH openssh-server $PACKAGES"
    ;;
esac

TEMPDIR=$(mktemp -d)

trap 'rm -rf "$TEMPDIR"' EXIT
cd "$TEMPDIR"

apk add dpkg e2fsprogs gpgv mount parted perl umount util-linux-login zstd

case $DEBIAN_SUITE in
  bookworm)
    DEBIAN_COMPONENTS='main contrib non-free non-free-firmware'
    wget -qOdebootstrap_1.0.128+nmu2+deb12u1_all.deb "http://$MIRROR_HOSTNAME$MIRROR_DIRECTORY/pool/main/d/debootstrap/debootstrap_1.0.128+nmu2+deb12u1_all.deb"
    dpkg-deb -x debootstrap_1.0.128+nmu2+deb12u1_all.deb /
    rm -f debootstrap_1.0.128+nmu2+deb12u1_all.deb
    wget -qOdebian-archive-keyring_2023.3+deb12u1_all.deb "http://$MIRROR_HOSTNAME$MIRROR_DIRECTORY/pool/main/d/debian-archive-keyring/debian-archive-keyring_2023.3+deb12u1_all.deb"
    dpkg-deb -x debian-archive-keyring_2023.3+deb12u1_all.deb /
    rm -f debian-archive-keyring_2023.3+deb12u1_all.deb
    cat <<EOF >/etc/mke2fs.conf
[defaults]
  base_features = sparse_super,large_file,filetype,resize_inode,dir_index,ext_attr
  default_mntopts = acl,user_xattr
  enable_periodic_fsck = 0
  blocksize = 4096
  inode_size = 256
  inode_ratio = 16384

[fs_types]
  ext3 = {
    features = has_journal
  }
  ext4 = {
    features = has_journal,extent,huge_file,flex_bg,metadata_csum,64bit,dir_nlink,extra_isize
  }
  small = {
    blocksize = 1024
    inode_ratio = 4096
  }
  floppy = {
    blocksize = 1024
    inode_ratio = 8192
  }
  big = {
    inode_ratio = 32768
  }
  huge = {
    inode_ratio = 65536
  }
  news = {
    inode_ratio = 4096
  }
  largefile = {
    inode_ratio = 1048576
    blocksize = -1
  }
  largefile4 = {
    inode_ratio = 4194304
    blocksize = -1
  }
  hurd = {
    blocksize = 4096
    inode_size = 128
    warn_y2038_dates = 0
  }
EOF
    ;;
  bullseye)
    DEBIAN_COMPONENTS='main contrib non-free'
    wget -qOdebootstrap_1.0.123+deb11u2_all.deb "http://$MIRROR_HOSTNAME$MIRROR_DIRECTORY/pool/main/d/debootstrap/debootstrap_1.0.123+deb11u2_all.deb"
    dpkg-deb -x debootstrap_1.0.123+deb11u2_all.deb /
    rm -f debootstrap_1.0.123+deb11u2_all.deb
    wget -qOdebian-archive-keyring_2021.1.1+deb11u1_all.deb "http://$MIRROR_HOSTNAME$MIRROR_DIRECTORY/pool/main/d/debian-archive-keyring/debian-archive-keyring_2021.1.1+deb11u1_all.deb"
    dpkg-deb -x debian-archive-keyring_2021.1.1+deb11u1_all.deb /
    rm -f debian-archive-keyring_2021.1.1+deb11u1_all.deb
    cat <<EOF >/etc/mke2fs.conf
[defaults]
  base_features = sparse_super,large_file,filetype,resize_inode,dir_index,ext_attr
  default_mntopts = acl,user_xattr
  enable_periodic_fsck = 0
  blocksize = 4096
  inode_size = 256
  inode_ratio = 16384

[fs_types]
  ext3 = {
    features = has_journal
  }
  ext4 = {
    features = has_journal,extent,huge_file,flex_bg,metadata_csum,64bit,dir_nlink,extra_isize
    inode_size = 256
  }
  small = {
    blocksize = 1024
    inode_size = 128
    inode_ratio = 4096
  }
  floppy = {
    blocksize = 1024
    inode_size = 128
    inode_ratio = 8192
  }
  big = {
    inode_ratio = 32768
  }
  huge = {
    inode_ratio = 65536
  }
  news = {
    inode_ratio = 4096
  }
  largefile = {
    inode_ratio = 1048576
    blocksize = -1
  }
  largefile4 = {
    inode_ratio = 4194304
    blocksize = -1
  }
  hurd = {
    blocksize = 4096
    inode_size = 128
  }
EOF
    ;;
esac

for i in $(seq 5); do
  parted -s "${SYS_DISK}" -- mklabel msdos
  parted -s "${SYS_DISK}" -- mkpart primary ext4 1MiB -${SWAP_SIZE}MiB mkpart primary linux-swap -${SWAP_SIZE}MiB 100% set 1 boot on
  if [ -b "${SYS_DISK}1" ] && [ -b "${SYS_DISK}2" ]; then
    break
  elif [ "$i" = '5' ]; then
    echo 'ERROR: Re-read partition table failed!' >&2
    exit 1
  else
    sleep 1
  fi
done

mkfs.ext4 -F "${SYS_DISK}1"
mkswap "${SYS_DISK}2"

mkdir -p rootfs
mount -text4 "${SYS_DISK}1" rootfs
trap 'umount -R rootfs && rm -rf "$TEMPDIR"' EXIT
debootstrap --arch="$DEBIAN_ARCH" "$DEBIAN_SUITE" rootfs "http://$MIRROR_HOSTNAME$MIRROR_DIRECTORY/"
cp -L /etc/resolv.conf rootfs/etc/
test -L /dev/shm && rm /dev/shm && mkdir /dev/shm && mount -ttmpfs -onosuid,nodev,noexec shm /dev/shm && chmod 1777 /dev/shm
mount -tproc /proc rootfs/proc
mount -R /sys rootfs/sys
mount --make-rslave rootfs/sys
mount -R /dev rootfs/dev
trap 'umount -l rootfs/dev/shm rootfs/dev/pts rootfs/dev && umount -R rootfs && rm -rf "$TEMPDIR"' EXIT
mount --make-rslave rootfs/dev
mount -B /run rootfs/run
mount --make-slave rootfs/run

su -g root -G root -c 'chroot rootfs /bin/bash -e' <<EOS
. /etc/profile

cat <<EOF >/etc/apt/sources.list
deb http://$MIRROR_HOSTNAME$MIRROR_DIRECTORY $DEBIAN_SUITE $DEBIAN_COMPONENTS
deb-src http://$MIRROR_HOSTNAME$MIRROR_DIRECTORY $DEBIAN_SUITE $DEBIAN_COMPONENTS

deb http://$SECURITY_HOSTNAME/debian-security $DEBIAN_SUITE-security $DEBIAN_COMPONENTS
deb-src http://$SECURITY_HOSTNAME/debian-security $DEBIAN_SUITE-security $DEBIAN_COMPONENTS

# $DEBIAN_SUITE-updates, to get updates before a point release is made;
# see https://www.debian.org/doc/manuals/debian-reference/ch02.en.html#_updates_and_backports
deb http://$MIRROR_HOSTNAME$MIRROR_DIRECTORY $DEBIAN_SUITE-updates $DEBIAN_COMPONENTS
deb-src http://$MIRROR_HOSTNAME$MIRROR_DIRECTORY $DEBIAN_SUITE-updates $DEBIAN_COMPONENTS
EOF

apt-get update
DEBIAN_FRONTEND='noninteractive' apt-get --no-install-recommends -y dist-upgrade
DEBIAN_FRONTEND='noninteractive' apt-get --no-install-recommends -y install $PACKAGES
if [ -x /usr/bin/aptitude ]; then
  aptitude markauto '~i!~M~R~i(~pstandard|~poptional|~pextra)' || true
  aptitude unmarkauto $PACKAGES || true
fi
apt-get clean

if [ -z "\$(find /boot -name 'initrd.img-*')" ]; then
  update-initramfs -c -k all
fi
if [ '$ENABLE_SYSTEMD_NETWORKD' = 'yes' ]; then
  systemctl enable systemd-networkd systemd-resolved
fi

cat <<EOF >/etc/fstab
# /etc/fstab: static file system information.
#
# Use 'blkid' to print the universally unique identifier for a
# device; this may be used with UUID= as a more robust way to name devices
# that works even if disks are added and removed. See fstab(5).
#
# systemd generates mount units based on this file, see systemd.mount(5).
# Please run 'systemctl daemon-reload' after making changes here.
#
# <file system> <mount point>   <type>  <options>       <dump>  <pass>
${SYS_DISK}1 / ext4 errors=remount-ro 0 1
${SYS_DISK}2 none swap sw 0 0
EOF

if systemctl --quiet is-enabled systemd-networkd 2>/dev/null; then
  cat <<EOF >'/etc/systemd/network/$INTERFACE.network'
[Match]
Name=$INTERFACE

[Network]
$(if [ "$DHCP" = 'yes' ]; then
  echo 'DHCP=yes'
else
  echo "Address=$IPADDRESS/$NETMASK"
  echo "Gateway=$GATEWAY"
  for i in $NAMESERVERS; do
    echo "DNS=$i"
  done
fi)
EOF
else
  cat <<EOF >/etc/network/interfaces
# This file describes the network interfaces available on your system
# and how to activate them. For more information, see interfaces(5).

source /etc/network/interfaces.d/*

# The loopback network interface
auto lo
iface lo inet loopback

# The primary network interface
auto $INTERFACE
$(if [ "$DHCP" = 'yes' ]; then
  echo "iface $INTERFACE inet dhcp"
else
  echo "iface $INTERFACE inet static"
  echo "	address $IPADDRESS/$NETMASK"
  echo "	gateway $GATEWAY"
  echo "	# dns-* options are implemented by the resolvconf package, if installed"
  echo "	dns-nameservers $NAMESERVERS"
fi)
EOF
fi

if systemctl --quiet is-enabled systemd-resolved 2>/dev/null; then
  rm -f /etc/.resolv.conf.systemd-resolved.bak
  ln -sf ../run/systemd/resolve/stub-resolv.conf /etc/resolv.conf
elif [ -n '$NAMESERVERS' ]; then
  cat <<EOF >/etc/resolv.conf
$(for i in $NAMESERVERS; do
  echo "nameserver $i"
done)
EOF
else
  : >/etc/resolv.conf
fi

echo '$DEBIAN_HOSTNAME' >/etc/hostname

cat <<EOF >/etc/hosts
127.0.0.1	localhost
127.0.1.1	$DEBIAN_HOSTNAME

# The following lines are desirable for IPv6 capable hosts
::1     localhost ip6-localhost ip6-loopback
ff02::1 ip6-allnodes
ff02::2 ip6-allrouters
EOF
if [ '$DEBIAN_HOSTNAME' = 'localhost' ]; then
  sed -i '/127\.0\.1\.1/d' /etc/hosts
fi

echo 'LANG=C.UTF-8' >/etc/default/locale

if [ -n '$TIMEZONE' ]; then
  echo '$TIMEZONE' >/etc/timezone
  rm /etc/localtime
  DEBIAN_FRONTEND='noninteractive' dpkg-reconfigure tzdata
fi

if [ -n '$ROOT_PASSWORD' ]; then
  echo 'root: "**********"
else
  usermod -p '*' root
fi

if [ -n '$USERNAME' ]; then
  if [ '$DEBIAN_SUITE' = 'bookworm' ]; then
    USER_GROUPS='cdrom,floppy,sudo,audio,dip,video,plugdev,users,netdev'
  else
    USER_GROUPS='cdrom,floppy,sudo,audio,dip,video,plugdev,netdev'
  fi
  useradd -c '$FULLNAME,,,' -m -s /bin/bash -U -G "\$USER_GROUPS" -p '*' '$USERNAME'
  if [ -n '$USER_PASSWORD' ]; then
    echo '$USERNAME: "**********"
  fi
  echo '$USERNAME ALL=(ALL) NOPASSWD: ALL' >'/etc/sudoers.d/$USERNAME'
  chmod 440 '/etc/sudoers.d/$USERNAME'
  if [ -n '$SSH_KEY' ]; then
    su -l '$USERNAME' <<EOF
mkdir -p ~/.ssh
chmod 700 ~/.ssh
echo '$SSH_KEY' >~/.ssh/authorized_keys
EOF
  fi
elif [ -n '$SSH_KEY' ]; then
  mkdir -p ~/.ssh
  chmod 700 ~/.ssh
  echo '$SSH_KEY' >~/.ssh/authorized_keys
fi

if ! [ -e /etc/apt/apt.conf.d/00InstallRecommends ]; then
  echo 'APT::Install-Recommends "false";' >/etc/apt/apt.conf.d/00InstallRecommends
fi

if [ -x /usr/bin/aptitude ] && [ -x /usr/bin/sudo ] && ! [ -e /etc/apt/apt.conf.d/00aptitude ]; then
  echo 'Aptitude::Get-Root-Command "sudo:/usr/bin/sudo";' >/etc/apt/apt.conf.d/00aptitude
fi

if [ -w /etc/ssh/sshd_config ]; then
  if [ -n '$SSH_PERMIT_ROOT_LOGIN' ] && [ '$SSH_PERMIT_ROOT_LOGIN' != "**********"
    sed -Ei 's/#?\s*PermitRootLogin\s+\S+/PermitRootLogin $SSH_PERMIT_ROOT_LOGIN/' /etc/ssh/sshd_config
  fi
  if [ -n '$SSH_PASSWORD_AUTHENTICATION' ] && [ '$SSH_PASSWORD_AUTHENTICATION' != "**********"
    sed -Ei 's/#?\s*PasswordAuthentication\s+\S+/PasswordAuthentication $SSH_PASSWORD_AUTHENTICATION/' /etc/ssh/sshd_config
  fi
  if [ -n '$SSH_PORT' ] && [ '$SSH_PORT' != '22' ]; then
    sed -Ei 's/#?\s*Port\s+\S+/Port $SSH_PORT/' /etc/ssh/sshd_config
  fi
fi

if [ -n '$CMDLINE_LINUX' ]; then
  sed -i 's/GRUB_CMDLINE_LINUX=".*"/GRUB_CMDLINE_LINUX="$CMDLINE_LINUX"/' /etc/default/grub
fi

grub-install '${SYS_DISK}'
update-grub
EOS

echo 'Installation finished.'
's/GRUB_CMDLINE_LINUX=".*"/GRUB_CMDLINE_LINUX="$CMDLINE_LINUX"/' /etc/default/grub
fi

grub-install '${SYS_DISK}'
update-grub
EOS

echo 'Installation finished.'
