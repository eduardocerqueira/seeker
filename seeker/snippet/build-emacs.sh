#date: 2025-01-09T16:56:24Z
#url: https://api.github.com/gists/86f6ed0c7cc6340b2194f607ce0b454d
#owner: https://api.github.com/users/namanhboi

#!/usr/bin/env bash

## Author: Abidán Brito
## This script builds GNU Emacs 29.1 with support for native elisp compilation,
## tree-sitter, libjansson (C JSON library), pure GTK and mailutils.

# Exit on error and print out commands before executing them.
set -euxo pipefail

# Let's set the number of jobs to something reasonable; keep 2 cores
# free to avoid choking the computer during compilation.
JOBS=`nproc --ignore=2`

# Clone repo locally and get into it.
git clone --depth 1 --branch emacs-29 git://git.savannah.gnu.org/emacs.git
pushd emacs

# Get essential dependencies.
sudo apt install -y build-essential \
    texinfo \
    libgnutls28-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff5-dev \
    libgif-dev \
    libxpm-dev \
    libncurses-dev \
    libgtk-3-dev \
    libtree-sitter-dev \
    libmagick++-dev \
    #libwebkit2gtk-4.1-dev \

# Get dependencies for gcc-10 and the build process.
sudo apt update -y
sudo apt install -y gcc-11 \
    g++-11 \
    libgccjit0 \
    libgccjit-11-dev \
    autoconf \

# Get dependencies for fast JSON.
sudo apt install -y libjansson4 libjansson-dev

# Get GNU Mailutils (protocol-independent mail framework).
sudo apt install -y mailutils

# Enable source packages and get dependencies for whatever
# Emacs version Ubuntu supports by default.
#
# Taken from here:
# https://www.masteringemacs.org/article/speed-up-emacs-libjansson-native-elisp-compilation
#sudo sed -i 's/# deb-src/deb-src/' /etc/apt/sources.list \
#    && apt update \
#    && apt build-dep -y emacs

# Stop debconf from complaining about postfix nonsense.
DEBIAN_FRONTEND=noninteractive

# Needed for compiling libgccjit or we'll get cryptic error messages.
export CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11

# Configure and run.
# NOTE(abi): binaries should go to /usr/local/bin by default.
#
# Options:
#    --with-native-compilation  ->  use the libgccjit native compiler
#    --with-pgtk                ->  better font rendering
#    --with-x-toolkit=gtk3      ->  widgets toolkit
#    --with-tree-sitter         ->  syntax parsing
#    --with-wide-int            ->  larger file size limit
#    --with-json                ->  fast JSON
#    --with-gnutls              ->  TLS/SSL
#    --with-mailutils           ->  e-mail
#    --without-pop              ->  no pop3 (insecure channels)
#    --with-cairo               ->  vector graphics backend
#    --with-imagemagick         ->  raster images backend
./autogen.sh \
    && ./configure \
    --with-native-compilation \
    --with-pgtk \
    --with-x-toolkit=gtk3 \
    --with-tree-sitter \
    --with-wide-int \
    --with-json \
    --with-modules \
    --without-dbus \
    --with-gnutls \
    --with-mailutils \
    --without-pop \
    --with-cairo \
    --with-imagemagick \
        
    # Other interesting compilation options:
    #
    #--prefix=""                    # output binaries location
    #--with-x-toolkit=lucid         # supposedly more stable
    #--with-xwidgets

    # Compiler flags:
    # -O2                   ->  Turn on a bunch of optimization flags. There's also -O3, but it increases
    #                           the instruction cache footprint, which may end up reducing performance.
    # -pipe                 ->  Reduce temporary files to the minimum.
    # -mtune=native         ->  Optimize code for the local machine (under ISA constraints).
    # -march=native         ->  Enable all instruction subsets supported by the local machine.
    # -fomit-frame-pointer  ->  Small functions don't need a frame pointer (optimization).
    #
    # https://lemire.me/blog/2018/07/25/it-is-more-complicated-than-i-thought-mtune-march-in-gcc/
    CFLAGS="-O2 -pipe -mtune=native -march=native -fomit-frame-pointer"

# Build.
# NOTE(abi): NATIVE_FULL_AOT=1 ensures native compilation ahead-of-time for all
#            elisp files included in the distribution.
make -j${JOBS} NATIVE_FULL_AOT=1 \
    && make install

# Return to the original path.
popd
