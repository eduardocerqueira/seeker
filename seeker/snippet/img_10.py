#date: 2022-06-15T17:13:51Z
#url: https://api.github.com/gists/8d5e324ed1f39fe924667ac339a50bbf
#owner: https://api.github.com/users/SoftSAR

from pgmagick.api import Image

img = Image('ouroku.jpg')
img.blur(10, 5)
img.write('ouroku_blur.jpg')