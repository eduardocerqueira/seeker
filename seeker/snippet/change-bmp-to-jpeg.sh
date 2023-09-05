#date: 2023-09-05T17:09:34Z
#url: https://api.github.com/gists/4132798ab22059a6efb2c926abd64b93
#owner: https://api.github.com/users/G00Z-G00Z

#!/bin/bash
# Pass in all files that you want to convert to jpg
# Example usage: ./change-bmp-to-jpeg.sh *.bmp

mogrify -format jpg $*
