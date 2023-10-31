#date: 2023-10-31T16:55:40Z
#url: https://api.github.com/gists/f0ed432fc9af9c55e8fb56eeff6bce80
#owner: https://api.github.com/users/claudeT31

import re
from collections import Counter



def letters_counter(my_text, ignore_special_characters=True):
    """
Compte le nombre d'occurrences de chaque lettre dans une chaîne de caractères, en ignorant les caractères spéciaux si spécifié.

Args:
    my_text: La chaîne de caractères à analyser.
    ignore_special_characters: Un booléen indiquant si les caractères spéciaux doivent être ignorés. Par défaut, True.

Returns:
    Une liste de tuples, où le premier élément de chaque tuple est une lettre et le second élément est le nombre de fois que cette lettre apparaît dans la chaîne de caractères.

Exemple:

>>> letters_counter("Hello @world !", ignore_special_characters=False)
dict_items([('H', 1), ('e', 1), ('l', 3), ('o', 2), ('w', 1), ('r', 1), ('d', 1), '@', 1], '!')
"""
    # Si l'argument ignore_special_characters est défini à True, supprime tous les caractères non alphanumériques de la chaîne de caractères.

    if ignore_special_characters:
        my_text = re.sub(r"[^\w]", "", my_text)

    # Crée une liste de toutes les lettres de la chaîne de caractères.

    letters = [letter for letter in my_text]

    # Compte le nombre d'occurrences de chaque lettre.

    resultat = Counter(letters)

    # Renvoie la liste des résultats.

    return resultat.items()

