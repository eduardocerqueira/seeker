#date: 2023-11-30T17:03:29Z
#url: https://api.github.com/gists/eefce1330e3d730580dc29625472d6be
#owner: https://api.github.com/users/robertoschwald

# Build Bareos Debian Packages, e.g. for Raspberry PI OS / arm64

BAREOS_BUILD_VERSION=23.0.0-pre
BAREOS_BUILD_DATE=`date --rfc-email`

git clone https://github.com/bareos/bareos.git
git checkout tags/WIP/${BAREOS_BUILD_VERSION}

sudo apt install dpkg-dev fakeroot dch
sudo apt build-dep ./bareos
cd bareos

# just compile
# mkdir build
# cd build
# cmake -Dclient-only=yes -Dconfdir=/etc/bareos ../
# make

# Only on initial build
cat <<EOF > debian/changelog
bareos (${BAREOS_BUILD_VERSION}) unstable; urgency=low

  * Initial package for ${BAREOS_BUILD_VERSION} release

 -- pi <packager@symentis.com>  ${BAREOS_BUILD_DATE}
EOF

# on subsequent builds, we increment the version in the changelog file and edit.
dch -i

# Build the bareos packages
fakeroot debian/rules clean
fakeroot debian/rules binary

# Alternatively
# dpkg-buildpackage --no-sign -b