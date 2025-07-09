#date: 2025-07-09T16:49:48Z
#url: https://api.github.com/gists/e638db412200476fdc8ff418f17ecd74
#owner: https://api.github.com/users/levihuayuzhang

#!/bin/zsh

# This script helps those who want to 
# get all Aerial wallpapers on macOS Sonoma
# all at once instead of clicking on settings one by one.

# Dependencies: jq, parallel
#   Install all dependencies via `brew install jq parallel`
#   After `parallel` installed run `sudo parallel --citation` first to read citation notice (IMPORTANT)

# The follwing script SHOULD be run under root user (IMPORTANT)

cd "/Library/Application Support/com.apple.idleassetsd/Customer" && \
cat entries.json | \
  jq -r '.assets[] | (.id + "," + .["url-4K-SDR-240FPS"])' | \
  parallel \
    wget \
      --no-check-certificate -q \
      -O './4KSDR240FPS/{= s:\,[^,]+$::; =}.mov' \
      '{= s:[^,]+\,::; =}';