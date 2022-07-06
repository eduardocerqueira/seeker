#date: 2022-07-06T17:09:31Z
#url: https://api.github.com/gists/d8bbb59789ef65112193c709dd1df99a
#owner: https://api.github.com/users/utkuozdemir

#!/usr/bin/env bash
set -euo pipefail

virsh list --all | \
awk '{print $2}' | \
xargs -L1 -I {} virsh destroy {} || true

virsh list --all | \
awk '{print $2}' | \
xargs -L1 -I {} virsh undefine {} || true


virsh vol-list --pool default | \
awk '{ print $1 }' | \
xargs -L1 -I {} virsh vol-delete --pool default {} || true

virsh net-list | \
awk '{print $1}' | \
xargs -L1 -I {} virsh net-destroy {} || true

vagrant global-status --prune || true
