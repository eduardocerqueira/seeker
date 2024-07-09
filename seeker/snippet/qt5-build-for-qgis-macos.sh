#date: 2024-07-09T16:58:15Z
#url: https://api.github.com/gists/1716e2bc5e4b3a6ac158bf2fd75c42eb
#owner: https://api.github.com/users/smellman

#!/bin/bash

# This script is used to build Qt5 for QGIS on macOS.
# see: https://github.com/Homebrew/homebrew-core/blob/HEAD/Formula/q/qt@5.rb

# Set the version of Qt5 to build
QT_VERSION=5.15.14
SRC_DIR=qt-everywhere-src-${QT_VERSION}
TAR_BALL=qt-everywhere-opensource-src-${QT_VERSION}.tar.xz
PREFIX=/opt/qt/${QT_VERSION}

# unarchive the tarball

tar jxf ${TAR_BALL}

# Change directory to the source directory

cd ${SRC_DIR}

# Configure qtwebengine start
## git clone qtwebengine

rm -fr qtwebengine
git clone https://code.qt.io/qt/qtwebengine.git qtwebengine
cd qtwebengine
git checkout -b v5.15.16-lts refs/tags/v5.15.16-lts
git submodule update --init --recursive

## patch qtwebengine
### libxml2 2.12 compatibility
cd src/3rdparty
curl "https://github.com/qt/qtwebengine-chromium/commit/c98d28f2f0f23721b080c74bc1329871a529efd8.patch?full_index=1" | patch -p 1
cd ../..

### python3 support for qtwebengine-chromium

cd src/3rdparty
curl "https://raw.githubusercontent.com/Homebrew/formula-patches/7ae178a617d1e0eceb742557e63721af949bd28a/qt5/qt5-webengine-chromium-python3.patch?full_index=1" | patch -p 1
cd ../..

### python3 support for qtwebengine

curl "https://raw.githubusercontent.com/Homebrew/formula-patches/a6f16c6daea3b5a1f7bc9f175d1645922c131563/qt5/qt5-webengine-python3.patch?full_index=1" | patch -p 1

### Use Debian patch to support Python 3.11

curl "https://raw.githubusercontent.com/Homebrew/formula-patches/9758d3dd8ace5aaa9d6720b2e2e8ea1b863658d5/qt5/qtwebengine-py3.11.patch" | patch -p 1

### Fix ffmpeg build with binutils

cd src/3rdparty/chromium/third_party/ffmpeg
curl "https://github.com/FFmpeg/FFmpeg/commit/effadce6c756247ea8bae32dc13bb3e6f464f0eb.patch?full_index=1" | patch -p 1
cd ../../../../..

### Use Gentoo's patch for ICU 74 support

curl "https://gitweb.gentoo.org/repo/gentoo.git/plain/dev-qt/qtwebengine/files/qtwebengine-6.5.3-icu74.patch?id=ba397fa71f9bc9a074d9c65b63759e0145bb9fa0" | patch -p 1

## patch qtwebengine end
cd ..

# configure qtwebengine end

# cataplut
git clone https://chromium.googlesource.com/catapult.git
cd catapult
git checkout 5eedfe23148a234211ba477f76fc2ea2e8529189
cd ..

# Fix build with Xcode 14.3.

cd qtlocation/src/3rdparty/mapbox-gl-native
curl "https://invent.kde.org/qt/qt/qtlocation-mapboxgl/-/commit/5a07e1967dcc925d9def47accadae991436b9686.diff" | patch -p 1
cd ../../../..

# Fix qmake with Xcode 15.

curl "https://raw.githubusercontent.com/Homebrew/formula-patches/086e8cf/qt5/qt5-qmake-xcode15.patch" | patch -p 1

# Fix qtmultimedia build with Xcode 15

curl "https://raw.githubusercontent.com/Homebrew/formula-patches/3f509180/qt5/qt5-qtmultimedia-xcode15.patch" | patch -p 1

# Fix use of macOS 14 only memory_resource on macOS 13

cd qtbase
curl "https://raw.githubusercontent.com/macports/macports-ports/56a9af76a6bcecc3d12c3a65f2465c25e05f2559/aqua/qt5/files/patch-qtbase-memory_resource.diff" | patch -p 0
cd ..

# CVE-2023-32763 Remove with Qt 5.15.15

cd qtbase
curl "https://invent.kde.org/qt/qt/qtbase/-/commit/deb7b7b52b6e6912ff8c78bc0217cda9e36c4bba.diff" | patch -p 1
cd ..

# CVE-2023-34410 Remove with Qt 5.15.15

cd qtbase
curl "https://invent.kde.org/qt/qt/qtbase/-/commit/2ad1884fee697e0cb2377f3844fc298207e810cc.diff" | patch -p 1
cd ..

# CVE-2023-37369 Remove with Qt 5.15.15

cd qtbase
curl "https://ftp.yz.yamagata-u.ac.jp/pub/qtproject/archive/qt/5.15/CVE-2023-37369-qtbase-5.15.diff" | patch -p 1
cd ..

# CVE-2023-38197 Remove with Qt 5.15.15

cd qtbase
curl "https://ftp.yz.yamagata-u.ac.jp/pub/qtproject/archive/qt/5.15/CVE-2023-38197-qtbase-5.15.diff" | patch -p 1
cd ..

# CVE-2023-51714 Remove with Qt 5.15.17

cd qtbase
curl "https://download.qt.io/official_releases/qt/5.15/0001-CVE-2023-51714-qtbase-5.15.diff" | patch -p 1
curl "https://download.qt.io/official_releases/qt/5.15/0002-CVE-2023-51714-qtbase-5.15.diff" | patch -p 1
cd ..


# patches end

# move catapult to qtwebengine
rm -fr qtwebengine/src/3rdparty/chromium/third_party/catapult
mv catapult qtwebengine/src/3rdparty/chromium/third_party/

# Configure the build

./configure -verbose -prefix ${PREFIX} -release -opensource -confirm-license -nomake examples -nomake tests -pkg-config -dbus-runtime -proprietary-codecs -no-rpath
make -j$(sysctl -n hw.logicalcpu)
make install
