#date: 2021-11-24T17:12:21Z
#url: https://api.github.com/gists/ceb184611b379e0686a68f19cecceafe
#owner: https://api.github.com/users/AnyKeyShik

#!/bin/sh

#####################################################
#   Install glibc with debug symbols on Arch Linux  #
#                                                   #
#         Make by AnyKeyShik Rarity (c) 2021        #
#####################################################

echo "Installing build dependencies"

# Install Dependencies
sudo pacman -S git svn gd lib32-gcc-libs patch make bison fakeroot devtools

echo "Checkout glibc sources from ArchLinux SVN"

# Checkout glibc source
svn checkout --depth=empty svn://svn.archlinux.org/packages && cd packages
svn update glibc && cd glibc/repos/core-x86_64

echo "Get user locales"

# Add current locale to locale.gen.txt
grep -v "#" /etc/locale.gen >> locale.gen.txt

echo "Change build config to debug"

# Enable debug build in PKGBUILD
sed -i 's#!strip#debug#' PKGBUILD

echo "Build libc"

# Build glibc and glibc-debug packages
makepkg --skipchecksums --config /usr/share/devtools/makepkg-x86_64.conf

echo "Installing libc"

# Install glibc-debug
sudo pacman -U *.pkg.tar.xz

echo "Update makepkg.conf"

sudo sed '/^OPTIONS/ s/!debug/debug/g; /^OPTIONS/ s/strip/!strip/g' /etc/makepkg.conf

echo "Installation complete!"