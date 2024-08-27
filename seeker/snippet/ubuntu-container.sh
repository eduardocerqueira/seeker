#date: 2024-08-27T16:52:24Z
#url: https://api.github.com/gists/52bf52fa78a0aec017cd3dcefb9df41f
#owner: https://api.github.com/users/Lokawn

#!/bin/bash -e
# Creates a systemd-nspawn container with Ubuntu

CODENAME=${CODENAME:-noble}

wget_or_curl () {
	if command -v wget >/dev/null; then
		wget "$1" -O "$2"
	elif command -v curl >/dev/null; then
		curl -L "$1" -o "$2"
	else
		echo "missing either curl or wget" >&2
		return 1
	fi
}

if [ $UID -ne 0 ]; then
	echo "run this script as root" >&2
	exit 1
fi

dest="$1"
if [ -z "$dest" ]; then
	echo "Usage: $0 <destination>" >&2
	exit 0
fi
if [ -e "$dest/usr/bin" ]; then
	echo "destination already seems to contain a root file system" >&2
	exit 1
fi

if [ "$(uname -m)" = x86_64 ]; then
	guestarch=amd64
elif [ "$(uname -m)" = aarch64 ]; then
	guestarch=arm64
else
	echo "unsupported architecture" >&2
	exit 1
fi
rootfs=$(mktemp)
trap 'rm $rootfs' EXIT

wget_or_curl "http://cloud-images.ubuntu.com/${CODENAME}/current/${CODENAME}-server-cloudimg-${guestarch}-root.tar.xz" $rootfs

mkdir -p "$dest"
tar -xaf $rootfs -C "$dest" --numeric-owner

sed '/^root: "**********"
rm "$dest/etc/resolv.conf" # systemd configures this
# https://github.com/systemd/systemd/issues/852
[ -f "$dest/etc/securetty" ] && \
	printf 'pts/%d\n' $(seq 0 10) >>"$dest/etc/securetty"
# container needs no mounts
>"$dest/etc/fstab"
# disable services and uninstall packages
systemd-nspawn -q -D "$dest" sh -c '
[ -s /etc/environment ] && . /etc/environment
for unit in ssh.service ssh.socket systemd-timesyncd systemd-networkd-wait-online systemd-resolved; do
    systemctl is-enabled "$unit" && systemctl disable "$unit"
done
apt-get -qq satisfy -y --purge "Conflicts: lxcfs, lxd, snapd, cloud-init" || \
apt-get -qq purge --autoremove snapd lxcfs lxd cloud-init
'


echo ""
echo "Ubuntu $CODENAME $guestarch container was created successfully"
ch container was created successfully"
