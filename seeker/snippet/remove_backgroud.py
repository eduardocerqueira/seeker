#date: 2024-08-26T17:10:50Z
#url: https://api.github.com/gists/f9493d323b256e167f7040fffdb2f9e4
#owner: https://api.github.com/users/alonsoir

from rembg import remove
from PIL import Image

## Path for input and output image
input_img = 'spiderman_with_bg.jpeg'
output_img = 'spiderman_without_bg.png'

## loading and removing background
inp = Image.open(input_img)
output = remove(inp)

## Saving background removed image to same location as input image
output.save(output_img)