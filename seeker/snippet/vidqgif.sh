#date: 2024-01-31T17:01:23Z
#url: https://api.github.com/gists/eb49286f26db36fa0ca544b63b462631
#owner: https://api.github.com/users/shandanjay

#!/bin/sh
# parameters: input, output

# make a 256-colors palette first
palettefull=$( mktemp --suffix=.png )
ffmpeg -i "$1" -vf 'palettegen' -y "$palettefull"

# quantize the palette (palettegen's builting limiter
# tries to preserve too much similar shades)
palettequant=$( mktemp --suffix=.png )
convert "$palettefull" -posterize 6 "$palettequant"

# initial compression
rawgif=$( mktemp --suffix=.gif )
ffmpeg -i "$1" -i "$palettequant" -lavfi "paletteuse=dither=0" -y "$rawgif"

# gifsicle optimization (the slowest stage)
gifsicle -O3 --lossy=80 "$rawgif" -o "$2"

# cleanup
rm "$palettefull" "$palettequant" "$rawgif"