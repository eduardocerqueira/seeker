#date: 2024-03-11T17:10:44Z
#url: https://api.github.com/gists/892d855586d9043a43b25d6a25e34d43
#owner: https://api.github.com/users/kevcrumb

#!/bin/sh
# Reproducibly build in a Split Linux container using the Docker method
# https://github.com/monero-project/monero/blob/master/contrib/gitian/DOCKRUN.md

# Abort on any error
set -e


# Official Monero Github can be used once PR #9129 is merged:
# https://github.com/monero-project/monero/pull/9129
#SOURCE=https://github.com/monero-project/monero.git
#BRANCH=
# Meanwhile you can build from kevcrumb's repository
SOURCE=https://github.com/kevcrumb/monero.git
BRANCH='gitian'


setup () {
        echo 'Setting up container ...'
        splt create ubuntu "$NAME" 234 default bionic amd64
        lxc-stop "$NAME"
        splt route "$NAME" leaky
        lxc-start "$NAME"
        splt vethup
}

prepare () {
        echo 'Configuring container ...'
        lxc-attach --clear-env --name "${NAME}" -- sh -c 'rm /etc/resolv.conf ; echo "nameserver 172.20.0.2" > /etc/resolv.conf'
        lxc-attach --clear-env --name "${NAME}" -- apt install -y docker.io apt-cacher-ng
        lxc-attach --clear-env --name "${NAME}" -- usermod -aG docker "$NAME"
}

checkout () {
        echo 'Clone a minimal Monero repo (for flaky connections, i.e. Tor)'
        lxc-attach --clear-env --name "${NAME}" -- su "$NAME" -c "cd ~ && git clone --branch ${BRANCH:-$VERSION} --depth 1 $SOURCE"
}

patch () {
        # TODO test this --- and/or simply PR
        lxc-attach --clear-env --name "${NAME}" -- sed -i 's|^  git clone \(https:\/\/github.com\/monero-project\/monero.*\)|  git clone -v --branch $VERSION --depth 1 \1|g' "/home/$NAME/monero/contrib/gitian/dockrun.sh"
        lxc-attach --clear-env --name "${NAME}" -- grep -Fw 'monero-project/monero' "/home/$NAME/monero/contrib/gitian/dockrun.sh"

        # TODO test this --- and/or simply PR
        # Is supposed to make the fetch smaller and thus the download less likely to fail, but seems to cause error:
        # "Submodule 'external/miniupnp' is not up-to-date."
        lxc-attach --clear-env --name "${NAME}" -- sed -i "s|bin\/gbuild', '-j',|bin/gbuild', '--skip-fetch', '-j',|g" "/home/$NAME/monero/contrib/gitian/gitian-build.py"
        lxc-attach --clear-env --name "${NAME}" -- grep -Fw 'gbuild' "/home/$NAME/monero/contrib/gitian/gitian-build.py"

        # TODO build only for the current architecture
        # NOTE that this sed apparently doesn't actually affect what is being built
        #lxc-attach --clear-env --name "${NAME}" -- sed -i "s|^  HOSTS=\".*\(`uname -m | cut -c1-3`[^ ]\+\).*\"|  HOSTS=\"\1\"|g" "/home/$NAME/monero/contrib/gitian/gitian-linux.yml"
        #lxc-attach --clear-env --name "${NAME}" -- grep '^  HOSTS=' "/home/$NAME/monero/contrib/gitian/gitian-linux.yml"
}

build () {
        # See monero/contrib/gitian/gitian-build.py for operating systems to choose from
        #OS=${OS:-'lafwm'} # Linux, Android, FreeBSD, Windows, MacOS
        OS=${OS:-'l'}
        THREADS=$(( $(nproc)-1 ))
        lxc-attach --clear-env --name "${NAME}" -- \
                su "$NAME" -c \
                "cd ~/monero/contrib/gitian/ && GH_USER='$GH_USER' OPT='-j $THREADS --os $OS' ./dockrun.sh $VERSION"
}

create_container_and_build () {
        setup
        
        if lxc-attach --clear-env --name "${NAME}" -- grep -q '^ID=ubuntu$' /etc/os-release ; then
                prepare && checkout && patch &&
                build
        fi
}


readonly GH_USER=$1 && shift # e.g. kevcrumb
readonly NAME=${1:-gitian-$(date +%Y%m%d%H%M%S.%N)}
readonly VERSION="${2:-v0.18.3.2}"

create_container_and_build