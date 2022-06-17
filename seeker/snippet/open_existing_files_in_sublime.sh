#date: 2022-06-17T16:55:57Z
#url: https://api.github.com/gists/899f91990001b7bc9cb8853cfc0d180d
#owner: https://api.github.com/users/zelaznik

#!/bin/bash

function find_existing_files() {
  # This prevents me from shooting myself in the foot when I'm
  # grepping for a pattern and forget to include "-l" before
  # using xargs to open the files in sublime

  for filepath in $@; do
    if [ -f $filepath ]; then
      echo $filepath
    fi
    if [ -d $filepath ]; then
      echo $filepath
    fi
  done
}

/Applications/Sublime\ Text.app/Contents/SharedSupport/bin/subl $(find_existing_files $@)
