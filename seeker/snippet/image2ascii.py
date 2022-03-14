#date: 2022-03-14T17:07:47Z
#url: https://api.github.com/gists/45ef2df9e8e09189af53416ebd4d3f66
#owner: https://api.github.com/users/hungneox

from PIL import Image

img = Image.open("star.jpg")
pixels = img.rotate(90).load()

density = "Ñ@#W$9876543210?!abc;:+=-,._ "

for i in range(img.size[0]):
    for j in range(img.size[1]):
        r, b, g = pixels[i, j]
        avg = int(round((r + b + g) / 3))
        length = 28  # len(density)
        percent = avg / 255
        char_index = round((1 - percent) * length)
        print(density[char_index] + " ", end="")
    print("")
