#date: 2022-07-27T16:55:56Z
#url: https://api.github.com/gists/9d0d348657e5fd4d307520e836611724
#owner: https://api.github.com/users/theanurin

#
# To start this script from remote:
#
#   curl https://gist.githubusercontent.com/theanurin/9d0d348657e5fd4d307520e836611724/raw/install-flutter-sdks.sh | /bin/bash
#

mkdir --parents ~/opt/flutter
cd ~/opt/flutter/
for FLUTTER_VERSION in 3.0.5 2.10.5 2.8.0; do
  echo
  echo "Installing Flutter v${FLUTTER_VERSION} ..."
  curl --fail https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_${FLUTTER_VERSION}-stable.tar.xz \
    | tar -xJp
  mv flutter ${FLUTTER_VERSION}-x64
done