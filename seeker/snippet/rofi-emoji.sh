#date: 2023-11-29T17:07:51Z
#url: https://api.github.com/gists/840c907f02dfb491a150be4e459893d1
#owner: https://api.github.com/users/mmirus

#!/usr/bin/env bash
#
#   Use rofi to pick emoji because that's what this
#   century is about apparently...
#
#   Requirements:
#     rofi, xsel, xdotool, curl
#
#   Usage:
#     1. Download all emoji
#        $ rofi-emoji --download
#
#     2. Run it!
#        $ rofi-emoji
#
#   Notes:
#     * You'll need a emoji font like "Noto Emoji" or "EmojiOne".
#     * Confirming an item will automatically paste it WITHOUT
#       writing it to your clipboard.
#     * Ctrl+C will copy it to your clipboard WITHOUT pasting it.
#

# Where to save the emojis file.
EMOJI_FILE="$HOME/.cache/emojis.txt"

function notify() {
    if [ "$(command -v notify-send)" ]; then
        notify-send "$1" "$2"
    fi
}


function download() {
    notify `basename "$0"` 'Downloading all emoji for your pleasure'

    curl https://unicode.org/emoji/charts/full-emoji-list.html |
        grep -Po "class='(chars|name)'>\K[^<]+" |
        paste - - > "$EMOJI_FILE"

    notify `basename "$0"` "We're all set!"
}


function display() {
    emoji=$(cat "$EMOJI_FILE" | grep -v '#' | grep -v '^[[:space:]]*$')
    line=$(echo "$emoji" | rofi -dmenu -i -p emoji -kb-custom-1 Ctrl+c $@)
    exit_code=$?

    line=($line)

    if [ $exit_code == 0 ]; then
        sleep 0.1  # Delay pasting so the text-entry can come active
        xdotool type --clearmodifiers "${line[0]}"
    elif [ $exit_code == 10 ]; then
        echo -n "${line[0]}" | xsel -i -b
    fi
}


# Some simple argparsing
if [[ "$1" =~ -D|--download ]]; then
    download
    exit 0
elif [[ "$1" =~ -h|--help ]]; then
    echo "usage: $0 [-D|--download]"
    exit 0
fi

# Download all emoji if they don't exist yet
if [ ! -f "$EMOJI_FILE" ]; then
    download
fi

# display displays :)
display
