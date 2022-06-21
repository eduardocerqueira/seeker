#date: 2022-06-21T17:03:41Z
#url: https://api.github.com/gists/9ac9306b08881f05d4365e128fe36b05
#owner: https://api.github.com/users/BigAlRender

#!/usr/bin/env bash
# exit on error
set -o errexit

if [[ ! -d $XDG_CACHE_HOME/yt-dlp ]]; then
  echo "...Downloading yt-dlp"
  cd $XDG_CACHE_HOME
  mkdir -p ./yt-dlp
  cd ./yt-dlp
  wget $(curl -s https://api.github.com/repos/yt-dlp/yt-dlp/releases/latest | jq -r '.assets[2] | .browser_download_url')
  chmod a+rx yt-dlp
  cd $HOME/project/src # Make sure we return to where we were
else
  echo "...Using yt-dlp from build cache"
fi
mkdir -p $HOME/project/src/yt-dlp
cp $XDG_CACHE_HOME/yt-dlp/yt-dlp HOME/project/src/yt-dlp/


# Add the rest of your build commands
# bundle install, etc.

# Either reference the binary directly:
# $XDG_CACHE_HOME/yt-dlp/bin/yt-dlp
#
# OR
#
# add it to the PATH as part of the start command/script:
# export PATH="$PATH:$XDG_CACHE_HOME/yt-dlp/bin