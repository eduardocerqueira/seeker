#date: 2024-12-17T17:08:22Z
#url: https://api.github.com/gists/541015bb1f8005fb000f2b638138d86a
#owner: https://api.github.com/users/zorky

from collections import namedtuple

BOLD = "\033[1m"
RESET = "\033[0m"

# Liste : mutable
print(f"\n{BOLD}1-{RESET} changement valeur élément d'une {BOLD}liste{RESET} possible : {BOLD}mutable{RESET}")
fruits = ["pomme", "banane", "cerise", "orange"]
fruits.append("orange")  # Liste modifiable
print(f"liste : {fruits} - set(liste) : {set(fruits)}\n")
fruits[0] = "pêche" # Element de la liste modifiable
print(f"le 1er élément est modifié : {fruits}\n")

# Tuple : immutable
coordonnees = (10, 20, 30, 30)
try:
    print(f"{BOLD}2-{RESET} changement valeur du 1er élément du {BOLD}tuple{RESET} impossible : {BOLD}immutable{RESET}")
    print(f"tuple : {coordonnees} - set(tuple) : {set(coordonnees)}")
    coordonnees[0] = 30 # Erreur, tuple immuable
except Exception as e:
    print(f"{e}\n")

# NamedTuple : immutable
print(f"{BOLD}3-a namedtuple{RESET} Personne")
Personne = namedtuple("Personne", ["nom", "prenom", "age"])
person1 = Personne(nom="Do", prenom="Jane", age=30)
print(f"{person1.nom}, {person1[1]}, {person1.age}\n")  # Accès par propriétés de l'objet namedtuple
print(f"{BOLD}3-b{RESET} changement valeur de la propriété prenom d'un objet {BOLD}namedtuple{RESET} impossible : {BOLD}immutable{RESET}")
try:
    person1.prenom = "John" # Erreur, namedtuple immuable
except Exception as e:
    print(e)

print(f"\n{BOLD}list{RESET} : mutable, ordonnée, éléments de types différents possible, doublons de valeurs possible (set permet de dédoublonner)")
print(f"{BOLD}tuple{RESET} : immutable, ordonné, éléments de types différents possible, doublons de valeurs possible (set permet de dédoublonner)")
print(f"{BOLD}namedtuple{RESET} : immutable, permet une meilleure lisibilité pour une structure simple car comme un objet avec ses propriétés")

# Exécution du script
#
# └─ $ ▶ python mutable-immutable.py
# 
# 1- changement valeur élément d'une liste possible : mutable
# liste : ['pomme', 'banane', 'cerise', 'orange', 'orange'] - set(liste) : {'orange', 'cerise', 'banane', 'pomme'}
# 
# le 1er élément est modifié : ['pêche', 'banane', 'cerise', 'orange', 'orange']
# 
# 2- changement valeur du 1er élément du tuple impossible : immutable
# tuple : (10, 20, 30, 30) - set(tuple) : {10, 20, 30}
# 'tuple' object does not support item assignment
# 
# 3-a namedtuple Personne
# Do, Jane, 30
# 
# 3-b changement valeur de la propriété prenom d'un objet namedtuple impossible : immutable
# can't set attribute
# 
# list : mutable, ordonnée, éléments de types différents possible, doublons de valeurs possible (set permet de dédoublonner)
# tuple : immutable, ordonné, éléments de types différents possible, doublons de valeurs possible (set permet de dédoublonner)
# namedtuple : immutable, permet une meilleure lisibilité pour une structure simple car comme un objet avec ses propriétés

