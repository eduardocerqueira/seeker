#date: 2024-08-29T17:01:50Z
#url: https://api.github.com/gists/a0b52ac833a4692bb0e9ab214f09afbe
#owner: https://api.github.com/users/kenvandine

#!/bin/bash

dir=$(dirname $(realpath $0))
in=$1

if [ $UID != 0 ];
then
	echo "Must be run with root privileges, for example with sudo"
	exit
fi

if [ $# -lt 1 ];
then
	echo "USAGE: sudo $0 SOURCE_ISO"
	exit
fi

if [ -d $dir/out ];
then
    rm $dir/out/* 2>/dev/null
else
    mkdir $dir/out
fi

if [ ! -d $dir/debs ];
then
    mkdir $dir/debs
fi

date=$(date "+%Y%m%d-%H%M")

# Output file should be NAME-UBUNTUVERSION-DATE-HOUR:MINUTE-ARCH.iso
out=$(echo "${in//ubuntu/NAME}")
out=$(echo "${out//base/$date}")

echo "Fetching local debian packages"
wget -O $dir/debs/google-chrome-stable_current_amd64.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb

cd $dir
echo $out > iso-version

echo "Creating $out"
echo "Adding local debs to pool"
livefs-editor $in out/repack.iso --add-debs-to-pool debs/*.deb
echo "Copying in autoinstall.yaml"
livefs-editor out/repack.iso out/repack2.iso --cp $PWD/autoinstall.yaml new/iso/autoinstall.yaml
rm -f out/repack.iso
livefs-editor out/repack2.iso out/repack3.iso --cp $PWD/iso-version new/iso/iso-version
rm -f out/repack2.iso
mv out/repack3.iso $out

echo "$out created"
