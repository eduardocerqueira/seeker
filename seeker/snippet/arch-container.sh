#date: 2024-08-27T16:52:24Z
#url: https://api.github.com/gists/52bf52fa78a0aec017cd3dcefb9df41f
#owner: https://api.github.com/users/Lokawn

#!/bin/bash -e
# Creates a systemd-nspawn container with Arch Linux

MIRROR=http://mirror.fra10.de.leaseweb.net/archlinux
ISO_DATE=latest
PKG_GROUPS="base"


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

[ "$(uname -m)" = x86_64 ] || { echo "unsupported architecture" >&2; exit 1; }
tarfile=$(mktemp)
trap 'rm $tarfile' EXIT

wget_or_curl "$MIRROR/iso/$ISO_DATE/archlinux-bootstrap-x86_64.tar.gz" $tarfile

mkdir -p "$dest"
tar -xzf $tarfile -C "$dest" --strip-components=1 --numeric-owner

# configure mirror
printf 'Server = %s/$repo/os/$arch\n' "$MIRROR" >"$dest"/etc/pacman.d/mirrorlist
sed '/^root: "**********"
rm "$dest/etc/resolv.conf" # systemd configures this
# https://github.com/systemd/systemd/issues/852
[ -f "$dest/etc/securetty" ] && \
	printf 'pts/%d\n' $(seq 0 10) >>"$dest/etc/securetty"
# seems to be this bug https://github.com/systemd/systemd/issues/3611
systemd-machine-id-setup --root="$dest"
# install the packages
systemd-nspawn -q -D "$dest" sh -c "
pacman-key --init && pacman-key --populate
pacman -Sy --noconfirm --needed ${PKG_GROUPS}
"


echo ""
echo "Arch Linux container was created successfully (bootstrapped from $ISO_DATE)"
ssfully (bootstrapped from $ISO_DATE)"
