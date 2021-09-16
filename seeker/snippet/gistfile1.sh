#date: 2021-09-16T16:59:49Z
#url: https://api.github.com/gists/24e76642678be979e634afabb458e8b7
#owner: https://api.github.com/users/kiyeto

$ brew install readline
==> Downloading http://ftpmirror.gnu.org/readline/readline-6.2.tar.gz
######################################################################## 100.0%
==> Downloading patches
######################################################################## 100.0%
==> Patching
patching file vi_mode.c
patching file callback.c
==> ./configure --prefix=/usr/local/Cellar/readline/6.2.1 --mandir=/usr/local/Cellar/readline/6.2.1/share/man --infodir=/usr/local/Cellar/readline/6.2.1/share/info --enable-multibyte
==> make install
==> Caveats
This formula is keg-only, so it was not symlinked into /usr/local.

OS X provides the BSD libedit library, which shadows libreadline.
In order to prevent conflicts when programs look for libreadline we are
defaulting this GNU Readline installation to keg-only.

Generally there are no consequences of this for you.
If you build your own software and it requires this formula, you'll need
to add its lib & include paths to your build variables:

    LDFLAGS  -L/usr/local/Cellar/readline/6.2.1/lib
    CPPFLAGS -I/usr/local/Cellar/readline/6.2.1/include
==> Summary
/usr/local/Cellar/readline/6.2.1: 28 files, 1.6M, built in 15 seconds

$ brew link readline
Linking /usr/local/Cellar/readline/6.2.1... 13 symlinks created
