#date: 2022-11-01T17:08:48Z
#url: https://api.github.com/gists/8c9a0b98ef95e58656f32a3f9ca447f0
#owner: https://api.github.com/users/pythoninthegrass

#!/usr/bin/env bash

# SOURCE: https://gist.github.com/Koze/2e1a9bf967b2bf865fc9

# close all windows (e.g., thousands of finder windows from unarchiver)
osascript -e 'tell application "Finder" to close windows'

## index 1 is frontmost window
# osascript 'tell application "Finder" to close window 1'