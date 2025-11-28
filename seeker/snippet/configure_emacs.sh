#date: 2025-11-28T16:51:55Z
#url: https://api.github.com/gists/9db185943e0acbd72f4c9bd5375c4b40
#owner: https://api.github.com/users/BabakSamimi

cd emacs
./configure --prefix=/opt/emacslite --program-suffix=-lite --without-all \
		  --with-native-compilation --with-zlib --with-threads --with-modules \
		  --with-dbus --with-gnutls --with-tree-sitter --with-libsystemd \
		  --with-pgtk --with-toolkit-scroll-bars --with-cairo --with-harfbuzz \
		  CFLAGS="-march=x86-64 -mtune=native -O2 -g -pipe -flto=auto -Wp,-D_FORTIFY_SOURCE=3 -Wformat -Werror=format-security -fstack-clash-protection -fcf-protection -ffile-prefix-map=/build/emacs/src=/usr/src/debug/emacs -flto=auto" \
		  LDFLAGS="-Wl,-O1 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,-z,pack-relative-relocs"
cd ..