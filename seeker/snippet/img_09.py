#date: 2022-06-15T17:11:10Z
#url: https://api.github.com/gists/1a933d68bb722239832d6523625e96ea
#owner: https://api.github.com/users/SoftSAR

from pgmagick.api import Image

img = Image('ouroku.jpg')
img.sharpen(1)
img.write('ouroku_sharpen1.jpg')