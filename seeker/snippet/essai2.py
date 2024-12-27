#date: 2024-12-27T16:45:30Z
#url: https://api.github.com/gists/51c99d40ba6874f382e22742740f0978
#owner: https://api.github.com/users/danicool1020

from math import *
from kandinsky import*
from random import*

emplacement_villes={"Paris" : (143,45), "New York" : (80,60), 
        "Mexico" : (60,95), "Moscou" : (170,43),
        "Tokyo" : (265,60)
}

def draw_circle(center_x, center_y, rayon, couleur):
    for x in range(center_x - rayon, center_x + rayon + 1):
        for y in range(center_y - rayon, center_y + rayon + 1):
            if (x - center_x)**2 + (y - center_y)**2 <= rayon**2:
                set_pixel(x, y, color)

center_x = emplacement_villes["Paris"][0]
center_y = emplacement_villes["Paris"][1]
rayon = 3
color = (255, 0, 0)

draw_circle(center_x, center_y, rayon, color)
draw_string("nom_de_ville",center_x + 10,center_y - 20,color)
emplacement_villes["Paris"][0]

liste_villes = list(emplacement_villes.keys())
index_aleatoire = randint(0, len(liste_villes) - 1)
ville_aleatoire = liste_villes[index_aleatoire]
center_x, center_y = emplacement_villes[ville]

rayon = 3
color = (255, 0, 0)
draw_circle(center_x, center_y, rayon, color)
draw_string(ville, center_x + 10, center_y - 20, color)
