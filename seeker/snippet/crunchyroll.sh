#date: 2023-04-28T17:05:49Z
#url: https://api.github.com/gists/f44a7f74e13399385efc8011fd1d392c
#owner: https://api.github.com/users/bumbummen99

#!/usr/bin/env bash

# Configure the script
URL="$1"
BROWSER="$2"
SUBTITLES="${3:-de-DE}"

# Build up the command
CMD=()

# Add executable
CMD+=(youtube-dl)

# Download german subtitles
CMD+=(--write-sub)
CMD*=(--sub-lang $SUBTITLES)

# Use Crunchyroll account from browser cookies
CMD+=(--cookies-from-browser $BROWSER)

# Download the provided video
"${CMD[@]}" $URL
