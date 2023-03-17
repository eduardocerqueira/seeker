#date: 2023-03-17T17:05:24Z
#url: https://api.github.com/gists/c91306b0206c783fb80905ca86910359
#owner: https://api.github.com/users/nilsandrey

#!/bin/sh
  ROTATION=$(shuf -n 1 -e '-' '')$(shuf -n 1 -e $(seq 0.05 .5))
  convert -density 150 $1 \
    -linear-stretch '1.5%x2%' \
    -rotate ${ROTATION} \
    -attenuate '0.01' \
    +noise  Multiplicative \
    -colorspace 'gray' $2