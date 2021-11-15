#date: 2021-11-15T17:14:21Z
#url: https://api.github.com/gists/7181561ef4ce10cab552f80b1c63fd4f
#owner: https://api.github.com/users/LalitaGertrudis

#!/bin/bash

ICON_PATH=$1

# Init Android icon setup
convert $ICON_PATH -resize 48x48 ./android/app/src/main/res/mipmap-mdpi/ic_launcher.png
convert $ICON_PATH -resize 72x72 ./android/app/src/main/res/mipmap-hdpi/ic_launcher.png
convert $ICON_PATH -resize 96x96 ./android/app/src/main/res/mipmap-xhdpi/ic_launcher.png
convert $ICON_PATH -resize 144x144 ./android/app/src/main/res/mipmap-xxhdpi/ic_launcher.png
convert $ICON_PATH -resize 192x192 ./android/app/src/main/res/mipmap-xxxhdpi/ic_launcher.png

convert $ICON_PATH -resize 48x48 -alpha set \
    \( +clone -distort DePolar 0 \
       -virtual-pixel HorizontalTile -background None -distort Polar 0 \) \
    -compose Dst_In -composite -trim +repage ./android/app/src/main/res/mipmap-mdpi/ic_launcher_round.png
convert $ICON_PATH -resize 72x72 -alpha set \
    \( +clone -distort DePolar 0 \
       -virtual-pixel HorizontalTile -background None -distort Polar 0 \) \
    -compose Dst_In -composite -trim +repage ./android/app/src/main/res/mipmap-hdpi/ic_launcher_round.png
convert $ICON_PATH -resize 96x96 -alpha set \
    \( +clone -distort DePolar 0 \
       -virtual-pixel HorizontalTile -background None -distort Polar 0 \) \
    -compose Dst_In -composite -trim +repage ./android/app/src/main/res/mipmap-xhdpi/ic_launcher_round.png
convert $ICON_PATH -resize 144x144 -alpha set \
    \( +clone -distort DePolar 0 \
       -virtual-pixel HorizontalTile -background None -distort Polar 0 \) \
    -compose Dst_In -composite -trim +repage ./android/app/src/main/res/mipmap-xxhdpi/ic_launcher_round.png
convert $ICON_PATH -resize 192x192 -alpha set \
    \( +clone -distort DePolar 0 \
       -virtual-pixel HorizontalTile -background None -distort Polar 0 \) \
    -compose Dst_In -composite -trim +repage ./android/app/src/main/res/mipmap-xxxhdpi/ic_launcher_round.png

echo "Android Icons created"