#date: 2022-07-27T17:15:39Z
#url: https://api.github.com/gists/b73e6557b8cf0e2a0a34bd004a992def
#owner: https://api.github.com/users/SoftAnnaLee

# Original method, using an image with the desired palette to map the colours to
# convert <source> -ordered-dither o4x4 -remap palette.png <output>

# New method where an image is created on the fly to act as the palette, and passed into conversion command.
# Taken from this StackOverflow answer; https://stackoverflow.com/a/25265526
convert -size 1x1 \
  xc:"#FFCDB1" \
  xc:"#FFB4A2" \
  xc:"#35989B" \
  xc:"#B5838D" \
  xc:"#6D6875" \
  -append txt:- |\
convert <source> -ordered-dither o4x4 -remap txt:- <output>