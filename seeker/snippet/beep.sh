#date: 2024-06-20T17:07:13Z
#url: https://api.github.com/gists/e41362c38a500067dfea935d3e3234b3
#owner: https://api.github.com/users/MateusBMP

#!/bin/bash
#
# Play a sound using the speaker
#
# Usage: ./play_sound.sh [time]
#     time: time in seconds to play the sound (default: 0.1)

if ! command -v speaker-test &> /dev/null
then
    echo "speaker-test could not be found"
    exit 1
fi

if [ $# -gt 1 ]
then
    echo "Usage: $0 [time]"
    echo "    time: time in seconds to play the sound (default: 0.1)"
    exit 1
fi

TIME=.1
if [ $# -eq 1 ]
then
    TIME=$1
fi

speaker-test -t sine -f 1000 -l 1 > /dev/null 2>&1 & sleep "$TIME" && kill -9 $!