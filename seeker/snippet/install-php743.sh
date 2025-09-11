#date: 2025-09-11T17:12:22Z
#url: https://api.github.com/gists/71031eee359a89c638e56bc59a68e271
#owner: https://api.github.com/users/samudraid

set -euo pipefail

# 0) (BRUTAL) singkirkan direktori ekstensi imap supaya tidak dibuild
[ -d ext/imap ] && mv ext/imap ext/imap.off || true

# 1) bersihkan build lama
[ -f Makefile ] && make distclean || true
rm -rf config.cache autom4te.cache configure.lineno aclocal.m4 libtool
unset CPPFLAGS CFLAGS LDFLAGS PKG_CONFIG_PATH

# 2) (jika ada) regen configure
[ -x ./buildconf ] && ./buildconf --force || true

# 3) configure ulang (tanpa IMAP) + FPM
./configure \
  --prefix=/usr/local/php74 \
  --with-bz2 \
  --enable-calendar \
  --with-curl \
  --enable-exif \
  --enable-ftp \
  --with-gettext \
  --enable-mbstring \
  --with-mysqli=mysqlnd \
  --with-pdo-mysql=mysqlnd \
  --enable-pcntl \
  --disable-posix \
  --with-readline \
  --enable-sockets \
  --enable-wddx \
  --with-xmlrpc \
  --with-xsl \
  --with-zlib \
  --disable-fileinfo \
  --enable-fpm \
  --without-imap --without-imap-ssl \
| tee ../configure.log

make -j"$(nproc)" | tee ../make.log
sudo make install
