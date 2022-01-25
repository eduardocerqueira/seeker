#date: 2022-01-25T16:51:38Z
#url: https://api.github.com/gists/76d74e72452c93870ff2bbc6c04c15a3
#owner: https://api.github.com/users/Trilex214

Chapitre 5 :

Fonction dérivée :

A) Définitions :

1. Rappel :
Une fonction f est dite dérivable en x0 si et seulement si 
limh->0 (f(x0+h)–f(x0))/h existe.

On note alors f'(x0) = limh->0 (f(x0+h)–f(x0))/h


2. Fonction dérivée :
Définition : Une fonction f est dite dérivable sur un intervalle I si et seulement si
pour tout x0 appartenant à I, f'(x0) existe.

La fonction définie sur I qui à x associe le nombre dérivé de f en x est appelée
( fonction ) dérivée de f. On la note f ' .
B) Dérivées des fonctions usuelles :
Fonction f

Ensemble de
définition de f

Fonction f '

Ensemble de
définition de f '

Fonction constante :
f x=k , k réel quelconque R

f ' x=0 pour tout x
réel. R

Fonction identité :
f x=x

R f '  x =1 R

Fonction inverse :
f x=
1
x

R* ( R\{0} ) f '  x =–
1
x
2 R* ( R\{0} )

Fonction racine carrée :
f x =x 

R+ ( [0;∞[ ) f '  x =
1
2 x 

R+* ( ]0;∞[ )

Fonction polynôme :
f x =x
k
; k ∈N.

R f ' x =kx k – 1 R

Fonction inverse de polynôme :
f (x)=
1
x
k
; k ∈N R* ( R\{0} )

−k
x
k+ 1 R* ( R\{0} )

Fonction cosinus :
f x=cosx 

R f '  x =–sinx  R

Fonction sinus :
f x=sinx 

R f '  x =cosx  R
Remarque : L'ensemble de définition de f n'est pas forcément l'ensemble de définition de f '.
On peut citer par exemple la fonction x . Une fonction définie sur un intervalle I n'est donc pas
forcément dérivable sur I.

Page 1/4

Lycée Français Prins Henrik Cours - 1ère Spé Maths 2021-2022

C) Opérations sur les fonctions dérivées :
1. Multiplication par une constante :

f étant une fonction dérivable sur I, k un réel quelconque, kf est dérivable sur I et kf '=kf ' .

2. Addition :

f et g étant des fonctions dérivables sur un intervalle I, f+g est une fonction dérivable sur I.
Sa dérivée est la somme des dérivées : fg'=f 'g ' .

3. Produit :

u et v étant des fonctions dérivables sur I, u.v est une fonction dérivable sur I.
Sa dérivée est u.v'=u' .vu.v ' .
Démonstration :
u.v' x  = lim
h 0
u xh.v xh –ux .v x

h

= lim
h 0
uxh.v xh –uxh.v x u xhv x –u x .v x

h

= lim
h 0
uxh[ v xh– v x ]v  x[uxh–ux]

h

= ux v ' x v  xu ' x 

Donc u.v'=u' .vu.v ' .
Cette formule permet de démontrer la formule de la dérivée d'une fonction polynôme f x=x
k
.

Cette démonstration utilise le principe de la récurrence, très utilisé en mathématiques.
La démonstration par récurrence consiste à montrer que la propriété est vraie au rang 1, ce qui
permet de supposer qu'elle est vraie au rang n. Il suffit alors de montrer que le passage du rang n au
rang n+1 fonctionne ce qui permet de conclure que la propriété est vraie pour tout n.
Démonstration par récurrence :
Soit la propriété P(n) : Soit la fonction f définie sur R par f x=x
n
, alors sa dérivée f ' x =nx n –1
.

P(1): { Soit la fonction f définie sur R par f x =x
1
, alors sa dérivée f '  x =1x1 – 1=1 } est vraie.

On suppose que { P(n) : Soit la fonction f définie sur R par f x=x
n
, alors sa dérivée

f ' x=nx n –1

} est vraie.
Montrons que P(n+1) est vraie.
x
n1=x×x
n
donc  x
n1
' = x×x
n
'=x× x
n
'x '×x

n=x×nxn – 11×x

n=nx nx

n=n1 x
n
.

Donc P(n+1) est vraie.
On en déduit que P(n) est vraie pour tout n appartenant à N.

Page 2/4

Lycée Français Prins Henrik Cours - 1ère Spé Maths 2021-2022

4. Inverse :
Soit v une fonction dérivable et n'admettant pas de racine sur I, 1
v
est dérivable sur I

et sa dérivée est 
1
v
'=–
v '
v
2
.

Démonstration :

lim
h 0
1
v xh
–
1
v x 
h

= lim
h 0
1
h

v x– v xh
v xh.v x 
 = lim
h 0
v  x –v xh
h
. lim
h 0
1
v  xh.v x 
= –
v '  x 
v  x
2

Donc 
1
v
'=–
v '
v
2

5. Quotient :
Soit deux fonctions u et v dérivables sur I, v ne s'annulant pas sur I, u/v est dérivable
sur I et sa dérivée est 
u
v
'=
u ' v –u.v '
v
2
.

Démonstration :
Elle se fait en constatant que u
v
=u×
1
v
et en appliquant les formules précédentes.

6. Fonction f(ax+b) :
Soit une fonction f dérivable sur un intervalle I telle que, pour tout x appartenant à I,
ax+b appartient à I, alors f axb est dérivable sur I et f axb'=af ' axb .

Page 3/4

Lycée Français Prins Henrik Cours - 1ère Spé Maths 2021-2022
D) Lien entre signe de la dérivée et variations de la fonction :
1. Signe de la dérivée d'une fonction monotone :

A) Fonction croissante :

Soit une fonction f définie, croissante et dérivable sur un intervalle I.
Par définition, pour tout x appartenant à I, f ' x = lim
h 0
f xh– f x 
h
.

Or f est croissante et xhx donc f xhf  x .
On en déduit que f ' x  est positive pour tout x appartenant à I.

B) Fonction décroissante :

Soit une fonction f définie, décroissante et dérivable sur un intervalle I.
Par définition, pour tout x appartenant à I, f ' x = lim
h 0
f xh– f x 
h
.

Or f est décroissante et xhx donc f xhf  x .
On en déduit que f ' x  est négative pour tout x appartenant à I.

2. Propriété :

Soit f une fonction définie et dérivable sur un intervalle I.

Si f ' x0 pour tout x appartenant à I alors f est croissante sur l'intervalle I.
Si f ' x0 pour tout x appartenant à I alors f est décroissante sur l'intervalle I.
Si f ' x=0 pour tout x appartenant à I alors f est constante sur l'intervalle I.

Remarque : Pour étudier les variations d'une fonction, on étudiera le signe de sa
dérivée.

3. Extrema :
Soit f une fonction définie et dérivable sur un intervalle I.
f admet un minimum ou un maximum local en x0 si et seulement si f ' x0
=0 .
