#date: 2021-09-15T16:59:42Z
#url: https://api.github.com/gists/08d5390325392bfe7b2257cced27d14a
#owner: https://api.github.com/users/diademiemi

convert ${INPUT} \
  -sharpen 0x5.0 \
  -modulate 100,150 \
  -resize 50% \
  -resize 200% \
  -contrast \
  +noise poisson \
  -equalize \
  -sharpen 0x4.0 \
  -fill red -tint 90 \
  -fill yellow -tint 100 \
  -gamma 0.5 \
  ${OUTPUT}