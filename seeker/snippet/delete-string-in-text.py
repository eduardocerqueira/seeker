#date: 2022-12-30T16:43:37Z
#url: https://api.github.com/gists/21520ef83f8f6bb04719d54ea83055cf
#owner: https://api.github.com/users/ladatascience

import regex as re

texte = "Drôle de texte"
chaine_a_retirer = "x"
nouveau_texte = re.sub(chaine_a_retirer, "", texte)

# nouveau_texte est à présent la chaîne de caractères "Drôle de tete".