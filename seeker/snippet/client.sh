#date: 2022-11-04T16:59:07Z
#url: https://api.github.com/gists/242295da3062ada94c4e8e7155975527
#owner: https://api.github.com/users/lukpueh

#!/bin/bash

###############################################################
# Demo client to updates root metadata with unrecognized fields
#
# Usage:
# 1. Install requirements: `pip install securesystemslib[crypto,pynacl] tuf`
# 2. Download client.sh and make executable
# 3. Run client: `./client.sh`

repo_url=https://gist.githubusercontent.com/lukpueh/242295da3062ada94c4e8e7155975527/raw/2f0ee5a9c37d940f1c3ab29ee135992fc1230439

# Download initial root with curl (out of band trust bootstrap)
curl ${repo_url}/1.root.json -o root.json

# Update root with TUF
#
# NOTE: This fails because the repo does not serve 'timestamp.json'
# The root update still works, even though the new root (version 2)
# includes an unknown field ("supported_versions").
python - << EOF
from tuf.ngclient import Updater

updater = Updater(
    metadata_dir=".",
    metadata_base_url="${repo_url}")

updater.refresh()
EOF