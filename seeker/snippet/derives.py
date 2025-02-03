#date: 2025-02-03T16:32:54Z
#url: https://api.github.com/gists/29520347c05e601ae903e535065b1504
#owner: https://api.github.com/users/natrixIsBack

from math import *
from kandinsky import *
from time import sleep

def derivee(f, x, h=1e-5):
    """Calcule la dérivée numérique de f en x."""
    return (f(x + h) - f(x)) / h

def main():
    clear()  # Nettoie l'écran
    set_pixel(0, 0, (255, 255, 255))  # Petite astuce pour forcer l'affichage sur NumWorks
    draw_string("Calcul de dérivée", 10, 10, (0, 0, 255))

    try:
        expr = input("Entrer la fonction f(x) : ")
        f = lambda x: eval(expr)  # Convertit l'entrée utilisateur en fonction
        x0 = float(input("Entrer x : "))
        
        resultat = derivee(f, x0)
        
        draw_string(f"f'({x0}) = {resultat}", 10, 40, (255, 0, 0))
        print(f"La dérivée en x = {x0} est {resultat}")
    except Exception as e:
        print("Erreur:", e)

main()

