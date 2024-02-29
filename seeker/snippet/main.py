#date: 2024-02-29T17:10:06Z
#url: https://api.github.com/gists/7b1bec6df111d83e561b51cfca36fe10
#owner: https://api.github.com/users/SArSERO

from PIL import Image
from PIL import ImageDraw


def gradient(n):
    g = 255 / 200
    new_image = Image.new("RGB", (200, 100), (0, 0, 0))
    draw = ImageDraw.Draw(new_image)
    h = 0
    if n == 'R':
        for i in range(200):
            draw.line((i, 0, i, 100), fill=(int(h), 0, 0), width=1)
            h = h + g
    elif n == 'G':
        for i in range(200):
            draw.line((i, 0, i, 100), fill=(0, int(h), 0), width=1)
            h = h + g
    else:
        for i in range(200):
            draw.line((i, 0, i, 100), fill=(0, 0, int(h)), width=1)
            h = h + g
    new_image.save("lines.png", "PNG")


gradient('R')
