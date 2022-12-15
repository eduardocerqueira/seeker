#date: 2022-12-15T16:41:51Z
#url: https://api.github.com/gists/d67f0f7591a52c74e17b37b301e38abe
#owner: https://api.github.com/users/gdamjan

#! /bin/bash
set -euo pipefail

VER=2.4.5
ARCH=amd64
URL="https://git.pleroma.social/pleroma/pleroma/-/jobs/artifacts/v$VER/download?job=$ARCH"
DESTDIR=/opt/pleroma-$VER
SYMLINK=/opt/pleroma

sudo mkdir $DESTDIR

curl -L "$URL" |
  sudo bsdtar -xvf - --strip-components 1 -C $DESTDIR --fflags

sudo ln -sfT $DESTDIR $SYMLINK
sudo systemctl restart pleroma
