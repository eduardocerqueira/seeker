#date: 2021-10-29T16:45:05Z
#url: https://api.github.com/gists/daac44d3e14e3e0879f526cde00037ca
#owner: https://api.github.com/users/Remi8404

from turtle import *
hideturtle()
def frac(longueur, n):
    if n == 0:
        forward(longueur)
    else:
        frac(longueur / 3, n - 1)
        left(108)
        frac(longueur / 3, n - 1)
        right(72)
        frac(longueur / 3, n - 1)
        right(72)
        frac(longueur / 3, n - 1)
        right(72)
        frac(longueur / 3, n - 1)
        left(108)
        frac(longueur / 3, n - 1)
    
def pentagone(taille, etape, r = 0 , g = 0, b = 0):
    color(r, g, b)
    frac(taille, etape)
    taille+=15
    g+=20
    color(r, g, b)
    right(72)
    frac(taille, etape)
    taille+=15
    g+=20
    color(r, g, b)
    right(72)
    frac(taille, etape)
    taille+=15
    g+=20
    color(r, g, b)
    right(72)
    frac(taille, etape)
    taille+=15
    g+=20
    color(r, g, b)
    right(72)
    frac(taille, etape)
    taille+=15
    g+=20
    color(r, g, b)
    right(72)
    frac(taille, etape)
    taille+=15
    g+=20
    color(r, g, b)
    right(72)
    frac(taille, etape)
    taille+=15
    g+=20
    color(r, g, b)
    right(72)
    frac(taille, etape)
    taille+=15
    g+=20
    color(r, g, b)
    right(72)
    frac(taille, etape)

penup();goto(0, 0);pendown();speed(10);


def spirale(repetition, longueur_init, r=0, g=0, b=0, etape=3):
    taille = longueur_init
    avancement_couleur = int(250/repetition)
    avancement_taille = int(100/repetition)
    for nombre in range(repetition):
        color(r, g, b)
        frac(taille, etape)
        taille+=avancement_taille
        g+=avancement_couleur
        right(int(72/repetition))

spirale(25, 10)
