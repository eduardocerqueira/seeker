#date: 2024-04-01T17:01:25Z
#url: https://api.github.com/gists/b5101b3ad378bcb6bc5c282349edfd4c
#owner: https://api.github.com/users/paolobarbolini

#!/bin/bash

# Read https://crates.io/data-access before using this.
# If going to do extensive number of queries consider cloning the database instead

USER_AGENT="LINK_TO_YOUR_GITHUB_PROFILE_OR_SOME_OTHER_CONTACT_METHOD"

for i in $(seq 1 200)
do
  curl \
    --http1.1 \
    --max-time 10 \
    --connect-timeout 5 \
    --retry 5 \
    --retry-connrefused \
    --fail \
    -H "User-Agent: $USER_AGENT" \
    "https://crates.io/api/v1/crates?page=$i&per_page=100&sort=recent-downloads" | jq -r '.crates[] | .name + "\t" + .max_version' >> crates.tsv

  sleep 3s
done
