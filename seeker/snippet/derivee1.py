#date: 2022-01-25T16:51:38Z
#url: https://api.github.com/gists/76d74e72452c93870ff2bbc6c04c15a3
#owner: https://api.github.com/users/Trilex214

from math import *
Chapitre 4 :

Nombres dérivés: dérivation locale.

I. Nombre dérivé :
1. Rappel(s) utile(s): taux de variation, coefficient directeur

Rappel : dans un repère orthonormé (O,I,J) le coefficient directeur de la droite (AB) est

a=(yb – ya)/(xb – xa)


Remarque: le coefficient directeur d'une droite n'existe que si et seulement si la droite (AB)
n'est pas parrallèle à l'axe des ordonnées.

Soit f une fonction definie sur un intervalle I, soient a et b deux nombres réels distincts
appartenant à I , le taux de variation de f entre a et b est donné par 

(f(b)– f(a))/(b – a)

Graphiquement Si A et B sont deux points de la courbe representive de f dans un repere
(O,I,J), alors le taux de variationde f entre a et b appelé également taux d'accroissement
de f entre a et b n'est autre que le coefficient directeur de la droite (AB).
Remarque: dans le cas d'une fonction affine le taux d'accroissement est une constante c'est le
coefficient directeur de la droite.
Exemples:

Remarque: On rencontre en physique nottament la notation 

Δ y/Δ x si on pose y = f(x).

2. Nombre dérivé d'une fonction f en un point A.

L'idée est de prendre deux points A et B sur la
courbe représentative de f.
En faisant tendre le point B vers le point A, on fait
tendre la droite (AB) vers une droite limite
particuliere: la tangente à Cf en A. On s'interesse
à son coefficient directeur.


Definition:
Soit f une fonction définie sur un intervalle I, soit a un réel appartenant à I, soit h un nombre
reel non nul tel que a+h appartienne egalement à I.
On dit que f est dérivable en a lorsque que le taux d'acroissement entre a et a+h de f tend
vers une valeur unique lorsque h tend vers zero. Cette valeur limite est appelée nombre dérivé
de f en a, il est noté f'(a).

Algebriquement:
Si les coordonnées de A sont (a;f (a)) et les coordonnées de B(a+h; f(a+h)) .
On cherche à déterminer f'(a) :
Le coefficient directeur de (AB) est 
(f(a+h)– f(a))/(a+h – a) = (f(a+h)– f (a))/h

Faire tendre B vers A correspond à faire tendre h vers 0. (ce que l'on ecrira: h–>0)
donc le coefficient de la droite (AB) tend vers f '(a) .

On a donc f'(a) = limh –>0 (f(a+h)– f(a))/h


Donc f'(2) = 4

3. Tangente à la courbe :
Définition:
Soit f une fonction derivable en a, soit A un point de la courbe de f de coordonnées A(a,f(a)),
La droite passant par A de coefficient directeur f'(a) est appelée tangente à la courbe
representative de f en A.
On travaille dans un repère (O,I,J). 
Soit A(xa;ya) 
L'équation de la tangente à Cf en A est y=f'(xa)x+ya−f'(xa)xa)

On peut la retenir ainsi : y−ya=f'(xa)(x−xa)
