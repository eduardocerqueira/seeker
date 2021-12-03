#date: 2021-12-03T16:58:17Z
#url: https://api.github.com/gists/756fd1062627ca1ea9f60923d161b0d8
#owner: https://api.github.com/users/mr-kenhoff

#!/bin/sh

export NDK=/home/$USER/Android/Sdk/ndk/23.1.7779620

export GMP_VERSION=6.2.1
export MPFR_VERSION=4.1.0
export XZ_VERSION=5.2.5
export GETTEXT_VERSION=0.21
export LIBICONV_VERSION=1.16
export ZLIB_VERSION=1.2.11
export LIBXML2_VERSION=2.9.12
export LIBQALCULATE_VERSION=3.22.0

# https://developer.android.com/ndk/guides/other_build_systems
export TOOLCHAIN=$NDK/toolchains/llvm/prebuilt/linux-x86_64
export SYSROOT=$TOOLCHAIN/sysroot
#export TARGET=aarch64-linux-android
export TARGET=armv7a-linux-androideabi
#export TARGET=i686-linux-android
#export TARGET=x86_64-linux-android
export API=21
export AR=$TOOLCHAIN/bin/llvm-ar
export CC=$TOOLCHAIN/bin/$TARGET$API-clang
export AS=$CC
export CXX=$TOOLCHAIN/bin/$TARGET$API-clang++
export LD=$TOOLCHAIN/bin/ld
export RANLIB=$TOOLCHAIN/bin/llvm-ranlib
export STRIP=$TOOLCHAIN/bin/llvm-strip

export LIB_ROOT=$PWD

rm -f gmp-$GMP_VERSION.tar.xz
rm -rf gmp-$GMP_VERSION
wget https://gmplib.org/download/gmp/gmp-$GMP_VERSION.tar.xz
tar -xf gmp-$GMP_VERSION.tar.xz
pushd gmp-$GMP_VERSION
./configure --host $TARGET
make
popd

rm -f mpfr-$MPFR_VERSION.tar.xz
rm -rf mpfr-$MPFR_VERSION
wget https://www.mpfr.org/mpfr-current/mpfr-$MPFR_VERSION.tar.xz
tar -xf mpfr-$MPFR_VERSION.tar.xz
pushd mpfr-$MPFR_VERSION
export C_INCLUDE_PATH="../gmp-$GMP_VERSION"
export LIBRARY_PATH="../gmp-$GMP_VERSION/.libs/"
./configure --host $TARGET --with-gmp-build=../gmp-$GMP_VERSION/
make
export C_INCLUDE_PATH=""
export LIBRARY_PATH=""
popd

rm -f xz-$XZ_VERSION.tar.xz
rm -rf xz-$XZ_VERSION
wget https://fossies.org/linux/misc/xz-$XZ_VERSION.tar.xz
tar -xf xz-$XZ_VERSION.tar.xz
pushd xz-$XZ_VERSION
./configure --host $TARGET
make
popd

rm -f libiconv-$LIBICONV_VERSION.tar.gz
rm -rf libiconv-$LIBICONV_VERSION
wget https://ftp.gnu.org/pub/gnu/libiconv/libiconv-$LIBICONV_VERSION.tar.gz
tar -xvzf libiconv-$LIBICONV_VERSION.tar.gz
pushd libiconv-$LIBICONV_VERSION
./configure --host $TARGET
make
popd

rm -f libxml2-$LIBXML2_VERSION.tar.gz
rm -rf libxml2-$LIBXML2_VERSION
wget ftp://xmlsoft.org/libxml2/libxml2-$LIBXML2_VERSION.tar.gz
tar -xvzf libxml2-$LIBXML2_VERSION.tar.gz
pushd libxml2-$LIBXML2_VERSION
./configure --host $TARGET --without-python --with-lzma=../xz-$XZ_VERSION
make
popd

#rm -f gettext-$GETTEXT_VERSION.tar.xz
#rm -rf gettext-$GETTEXT_VERSION
#wget https://mirrors.kernel.org/gnu/gettext/gettext-$GETTEXT_VERSION.tar.xz
#tar -xf gettext-$GETTEXT_VERSION.tar.xz
#pushd gettext-$GETTEXT_VERSION
#./configure --host $TARGET --disable-openmp
#make
#popd

rm -f libqalculate/$LIBQALCULATE_VERSION.tar.gz
rm -rf libqalculate/$LIBQALCULATE_VERSION
wget https://github.com/Qalculate/libqalculate/releases/download/v$LIBQALCULATE_VERSION/libqalculate-$LIBQALCULATE_VERSION.tar.gz
tar -xvzf libqalculate-$LIBQALCULATE_VERSION.tar.gz
pushd libqalculate-$LIBQALCULATE_VERSION
patch libqalculate/util.cc < ../liqalculate_util.patch
export CPPFLAGS="-I$LIB_ROOT/gmp-$GMP_VERSION -I$LIB_ROOT/mpfr-$MPFR_VERSION/src -I$LIB_ROOT/libiconv-$LIBICONV_VERSION/include"
export LDFLAGS="-L$LIB_ROOT/libiconv-$LIBICONV_VERSION/lib/.libs -L$LIB_ROOT/mpfr-$MPFR_VERSION/src/.libs -L$LIB_ROOT/gmp-$GMP_VERSION/.libs"
export LIBXML_CFLAGS="-I$LIB_ROOT/libxml2-$LIBXML2_VERSION/include"
export LIBXML_LIBS="-L$LIB_ROOT/libxml2-$LIBXML2_VERSION/"
./configure --host $TARGET --without-icu --without-libcurl --without-libintl-prefix --with-sysroot=$SYSROOT
make
popd
