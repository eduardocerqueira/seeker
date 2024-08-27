#date: 2024-08-27T16:52:24Z
#url: https://api.github.com/gists/52bf52fa78a0aec017cd3dcefb9df41f
#owner: https://api.github.com/users/Lokawn

#!/bin/bash -e
# Creates a systemd-nspawn container with Alpine

MIRROR=http://dl-cdn.alpinelinux.org/alpine
VERSION=${VERSION:-v3.20}
APKTOOLS_VERSION=2.14.4-r0


wget_or_curl () {
	if command -v wget >/dev/null; then
		wget -qO- "$1"
	elif command -v curl >/dev/null; then
		curl -Ls "$1"
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

if [[ "$(uname -m)" =~ ^i[3456]86|x86 ]]; then
	toolarch=x86
	guestarch=$toolarch
	[ "$(uname -m)" = x86_64 ] && guestarch=x86_64
elif [[ "$(uname -m)" =~ ^arm|aarch64 ]]; then
	toolarch=armv7
	guestarch=$toolarch
	[ "$(uname -m)" = aarch64 ] && guestarch=aarch64
else
	echo "unsupported architecture" >&2
	exit 1
fi
apkdir=$(mktemp -d)
trap 'rm -rf $apkdir' EXIT

wget_or_curl "$MIRROR/latest-stable/main/$toolarch/apk-tools-static-$APKTOOLS_VERSION.apk" \
	| tar -xz -C $apkdir || \
	{ echo "couldn't download apk-tools, the version might have changed..." >&2; exit 1; }

$apkdir/sbin/apk.static \
	-X $MIRROR/$VERSION/main -U --arch $guestarch \
	--allow-untrusted --root "$dest" \
	--initdb add alpine-base

mkdir -p "$dest"/{etc/apk,root}
# configure mirror
printf '%s/%s/main\n%s/%s/community\n' "$MIRROR" $VERSION "$MIRROR" $VERSION >"$dest"/etc/apk/repositories
for i in $(seq 0 10); do # https://github.com/systemd/systemd/issues/852
	echo "pts/$i" >>"$dest/etc/securetty"
done
# make console work
sed '/tty[0-9]:/ s/^/#/' -i "$dest"/etc/inittab
printf 'console::respawn:/sbin/getty 38400 console\n' >>"$dest"/etc/inittab
# minimal boot services
for s in hostname bootmisc syslog; do
	ln -s /etc/init.d/$s "$dest"/etc/runlevels/boot/$s
done
for s in killprocs savecache; do
	ln -s /etc/init.d/$s "$dest"/etc/runlevels/shutdown/$s
done


echo ""
echo "Alpine $VERSION $guestarch container was created successfully."
