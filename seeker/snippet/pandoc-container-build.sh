#date: 2022-05-09T16:52:06Z
#url: https://api.github.com/gists/c85b8105cb57ea37db363ffc5ccd0c2c
#owner: https://api.github.com/users/carlwgeorge

#!/usr/bin/bash
set -eu
buildah unshare << EOF
set -eu
ctr=\$(buildah from scratch)
mnt=\$(buildah mount \$ctr)
dnf \
    --releasever 35 \
    --disablerepo '*' \
    --enablerepo fedora,updates \
    --installroot \$mnt \
    --setopt 'tsflags=nodocs' \
    --setopt 'install_weak_deps=false' \
    --assumeyes \
    install coreutils-single glibc-minimal-langpack pandoc
dnf --installroot \$mnt clean all
rm -rf \$mnt/var/cache/dnf
buildah config --entrypoint '["/usr/bin/pandoc"]' \$ctr
buildah unmount \$ctr
buildah commit \$ctr localhost/pandoc
buildah rm \$ctr
EOF