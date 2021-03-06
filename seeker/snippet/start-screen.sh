#date: 2022-06-29T17:14:07Z
#url: https://api.github.com/gists/634d9d5e0b0e2f40c211d696763eb376
#owner: https://api.github.com/users/Dartegnian

#!/usr/bin/env bash

theme_location=$XDG_CONFIG_HOME/rofi/rofi-metro/start-screen.rasi
echo "" | rofi -theme $theme_location -show combi -combi-modi "drun" -combi-hide-mode-prefix
exit_code="$?"
echo $exit_code

case "$exit_code" in
    10)
        coproc ( thunderbird > /dev/null  2>&1 )
        ;;
    11)
        coproc ( firefox http:// > /dev/null  2>&1 )
        ;;
    12)
        coproc ( steam > /dev/null  2>&1 )
        ;;
    13)
        coproc ( discord > /dev/null  2>&1 )
        ;;
    14)
        coproc ( mpv > /dev/null  2>&1 )
        ;;
    *)
        ;;
esac
