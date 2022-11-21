#date: 2022-11-21T17:12:26Z
#url: https://api.github.com/gists/70c00cd3bd7d536875ad59846c6e74e2
#owner: https://api.github.com/users/BenjaminSantiago

background(0,255,255)
size(500, 500)

noStroke()
fill(255,0,255)

for x in range (0, width, 20):
    for y in range (0, height, 20):
        rect(x, y, 15, 15)