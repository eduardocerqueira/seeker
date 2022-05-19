#date: 2022-05-19T17:41:27Z
#url: https://api.github.com/gists/5647ad291eb7ed355ec52db0eedb013f
#owner: https://api.github.com/users/rubo77

#!/bin/bash
# siehe https://wiki.freifunk.net/ECDSA_Util

# dies gilt nur für Kiel 

# falls du noch keinen key hast:
# ecdsautil generate-key > freifunk_ruben_ecdsautil_key.secret
cat ecdsautil_key.secret | ecdsautil show-key > ecdsautil_key.public

branch=stable
#branch=rc
#mirror='http://[fda1:384a:74de:4242::2]/firmware/'$branch'/sysupgrade/'
mirror='https://freifunk.in-kiel.de/firmware/'$branch'/sysupgrade/'

wget $mirror"$branch.manifest.clean"

cp $branch.manifest.clean $branch.manifest
echo "---" >>$branch.manifest
ecdsasign $branch.manifest.clean < freifunk_ruben_ecdsautil_key.secret >>$branch.manifest
