#date: 2022-07-28T17:19:38Z
#url: https://api.github.com/gists/4c00f41c01b77e7e6dd6713135851cb1
#owner: https://api.github.com/users/h0tw1r3

#!/bin/bash

# halt on any errors
set -o pipefail -e

BREW_INSTALL_PATH=/home/linuxbrew/.linuxbrew

if ! [ -x "/home/linuxbrew/.linuxbrew/bin/brew" ] ; then
        if ! sudo -l -U root >/dev/null 2>&1 ; then
                echo "root sudo permission required to install"
                exit 2
        fi

        echo "=== Installing homebrew"
        if ! [ -f "${BREW_INSTALL_PATH}" ] ; then
                echo "=== Installing required packages"
                sudo yum -y groupinstall 'Development Tools'
                sudo yum -y install procps-ng curl file git
                sudo yum -y install libxcrypt-compat || true
                sudo mkdir -p "${BREW_INSTALL_PATH}"
                sudo chown -R "${USER}:" "${BREW_INSTALL_PATH}"
        fi
        cd /home/linuxbrew/.linuxbrew
        echo "=== Downloading Homebrew from github.com"
        curl -L https://github.com/Homebrew/brew/tarball/master | tar xz --strip 1 -C .
        eval "$(bin/brew shellenv)"
        echo "=== Updating Homebrew"
        brew update --force --quiet
else
        eval "$("${BREW_INSTALL_PATH}"/bin/brew shellenv)"
fi

set +o pipefail +e