#date: 2025-02-03T16:32:54Z
#url: https://api.github.com/gists/29520347c05e601ae903e535065b1504
#owner: https://api.github.com/users/natrixIsBack

from math import sqrt
from kandinsky import draw_string, clear

def solve_quadratic(a, b, c):
    """Résout une équation quadratique ax² + bx + c = 0."""
    delta = b**2 - 4*a*c

    if delta > 0:
        x1 = (-b - sqrt(delta)) / (2*a)
        x2 = (-b + sqrt(delta)) / (2*a)
        return f"x1 = {x1:.4f}, x2 = {x2:.4f}"
    elif delta == 0:
        x = -b / (2*a)
        return f"x = {x:.4f}"
    else:
        return "Pas de solution réelle"

def main():
    clear()
    draw_string("Résolution quadratique", 10, 10, (0, 0, 255))

    try:
        a = float(input("Entrer a : "))
        b = float(input("Entrer b : "))
        c = float(input("Entrer c : "))

        if a == 0:
            draw_string("a ne peut pas être 0", 10, 40, (255, 0, 0))
            print("Erreur : a ne peut pas être 0")
            return

        result = solve_quadratic(a, b, c)
        draw_string(result, 10, 40, (255, 0, 0))
        print(result)

    except Exception as e:
        print("Erreur:", e)

main()
# This comment was added automatically to allow this file to save.
# You'll be able to remove it after adding text to the file.
