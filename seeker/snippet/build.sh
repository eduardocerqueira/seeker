#date: 2023-02-02T17:06:18Z
#url: https://api.github.com/gists/4965e2a56650b5d8bb7b2716d6a6406b
#owner: https://api.github.com/users/mperon

#!/usr/bin/env bash

# you must have BREW installed!!!
if [[ -d /opt/local/bin ]]; then
  sudo mkdir -p /opt/local/{bin,lib,include,share}
  sudo chown $(whoami) /opt/local/{bin,lib,include,share}
  sudo chown -R $(whoami) /usr/local
fi

#sets some variables
[[ -z "$HOMEBREW_PREFIX" ]] && export HOMEBREW_PREFIX=$(brew --prefix)
BUILD_PREFIX=/opt/local


# first of all, install all dependencies
brew tap bell-sw/liberica
brew install --cask liberica-jdk19-full
brew install ant afflib libewf libpq
brew install afflib
brew install zlib

# sets java home
[[ -z "$JAVA_HOME" ]] && export JAVA_HOME=$(/usr/libexec/java_home 2> /dev/null)

#prepare things
brew install autoconf gettext automake gdbm gmp libksba libtool libyaml \
    openssl@1.1 pkg-config readline libtool
# fix libtoolnize and build apps..
[[ ! -f "$BUILD_PREFIX/bin/libtoolize" ]] && \
    ln -s $HOMEBREW_PREFIX/bin/glibtoolize $BUILD_PREFIX/bin/libtoolize
for f in aclocal autoconf autoheader automake autopoint autoreconf \
    libtoolize pkg-config; do
    [[ ! -f "$BUILD_PREFIX/bin/$f" ]] &&
        ln -s $HOMEBREW_PREFIX/bin/$f $BUILD_PREFIX/bin/$f
done

# now compile two libs
for n in libvhdi libvmdk; do
    git clone git@github.com:libyal/${n}.git
    pushd $PWD/$n
        ./synclibs.sh
        ./autogen.sh
        ./configure --prefix=$BUILD_PREFIX --enable-python --with-pyprefix
        make && make install
    popd
done
export PATH="/opt/homebrew/opt/openssl@1.1/bin:$PATH"
export LDFLAGS="-L/opt/homebrew/opt/openssl@1.1/lib $LDFLAGS"
export CPPFLAGS="-I/opt/homebrew/opt/openssl@1.1/include $CPPFLAGS"
export PKG_CONFIG_PATH="/opt/homebrew/opt/openssl@1.1/lib/pkgconfig:$PKG_CONFIG_PATH"

# https://slo-sleuth.github.io/tools/InstallingAutopsyOnMacOS.html#building-the-sleuthkit
sudo rm -f /usr/local/opt/openjdk
sudo mkdir -p /usr/local/opt/
sudo ln -s $JAVA_HOME /usr/local/opt/openjdk

# get correct sleuthkit for iped
git clone -b release-4.11.1_iped_patch https://github.com/sepinf-inc/sleuthkit
pushd $PWD/sleuthkit
./bootstrap
./configure --prefix=$BUILD_PREFIX
make && make install
popd