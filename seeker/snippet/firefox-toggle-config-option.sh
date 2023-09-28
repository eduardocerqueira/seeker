#date: 2023-09-28T16:57:38Z
#url: https://api.github.com/gists/5b94a22e48d1ff0528c527ffaae93d2a
#owner: https://api.github.com/users/simonLeary42

#!/bin/bash

set -eo pipefail

SETTING="mousewheel.default.delta_multiplier_y"
CONFIG_FILE_PATH="/home/simon/.mozilla/firefox/shlbuz9i.default-esr/prefs.js"
LOW="30" # comfortable with my macbook 2015 touchpad
HIGH="101" # 100 is default, firefox deletes default values from the file

firefox_found=$(pgrep -U "$(whoami)" -fx '(.*/)?firefox(-esr)?$' | uniq || true)

if [ ! -z "$firefox_found" ]; then
    echo "firefox is running! This won't work if firefox is running."
    echo "To kill all your firefox processes, press enter. Else, press Ctrl+C."
    read confirm_ok
    for firefox_pid in $firefox_found; do
        kill "$firefox_pid"
    done
fi

occurrences=$(grep -c "$SETTING" "$CONFIG_FILE_PATH" || true)
if [ "$occurrences" == "0" ]; then
    echo "scroll sentivity not found in config file!"
    echo "setting value to high ($HIGH)..."
    echo "user_pref(\"$SETTING\", $HIGH);" >> "$CONFIG_FILE_PATH"
    exit 0
fi
if [ "$occurrences" -gt 1 ]; then
    echo "something went wrong, scroll sentivity was found multiple times in the file."
    grep --color "$SETTING" "$CONFIG_FILE_PATH"
    exit 1
fi

setting_found=$(grep "$SETTING" "$CONFIG_FILE_PATH" | sed -E "s/user_pref\(\"$SETTING\", (.*?)\);/\1/")
if [ "$setting_found" == "$LOW" ]; then
    echo "switching from low ($LOW) to high ($HIGH)"
    sed -i -E "s/user_pref\(\"$SETTING\", $LOW\);/user_pref(\"$SETTING\", $HIGH);/" "$CONFIG_FILE_PATH"
    exit 0
fi
if [ "$setting_found" == "$HIGH" ]; then
    echo "switching from high ($HIGH) to low ($LOW)"
    sed -i -E "s/user_pref\(\"$SETTING\", $HIGH\);/user_pref(\"$SETTING\", $LOW);/" "$CONFIG_FILE_PATH"
    exit 0
fi
echo "setting found isn't my preconfigured low ($LOW) or high ($HIGH)"
echo "defaulting to high ($HIGH)"
sed -i -E "s/user_pref\(\"$SETTING\", $setting_found\);/user_pref(\"$SETTING\", $HIGH);/" "$CONFIG_FILE_PATH"
