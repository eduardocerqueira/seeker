#date: 2022-05-13T17:21:18Z
#url: https://api.github.com/gists/64be94df12b1f9078b0b64cbe5ef32f2
#owner: https://api.github.com/users/RegularRabbit05

from PIL import Image 

img = Image.open(r"ball.png") 

for i in range(60):
    n = img.rotate(360/60*i)
    name = "fold/"+str(i)+"rot.png"
    n.save(name)

