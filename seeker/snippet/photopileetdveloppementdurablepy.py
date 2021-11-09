#date: 2021-11-09T17:13:13Z
#url: https://api.github.com/gists/6e3990bc8c8c62ebad7019ebf1c5e167
#owner: https://api.github.com/users/REETROO

from math import *
Photopile et développement durable – correction
1/ une photopile convertie l’énergie lumineuse et énergie 
électrique (et en chaleur cause de la résistance interne)

2/ voir schéma ci-contre

3/ on a un circuit série (une maille) donc 
UPN = R x I donc I = UPN / R avec R variable donc si R
varie I varie, pour augmenter I on diminuera R
P = U x I = 50 x 4,5 = 225 W

4/ si on augmente le courant, l’intensité 
augmente donc les pertes (P = RxI²) augmentent

5/ graphiquement on trouve Pmax = 300 mW, ce qui
correspond une intensité : I= 80 mA

6/ rendement = ce que l’on récupère / ce que l’on donne
donc dans notre cas :
rendement = énergie électrique / énergie lumineuse
η = Eélec / Elum = Pélec x t / Plum x t = Pélec / Plum = 0,3 x 0,08 / 3
η = 0.008 = 0,8 %

7/ un rendement énergétique ne peut pas être = 1 puisque
lors de la conversion il y a toujours apparition de chaleur
donc de perte, la lumière n’est pas toujours perpendiculaire
au panneau il y a donc aussi une perte par réflexion
remarque : certain parc photovoltaïque se sont installés sur
des plan d’eau pour justement récupérer cette chaleur et la
« stocker » dans l’eau (chauffage au sol par exemple)

8/ l’immeuble se situe Valence (zone 3) sur la ligne 1530
donc à Valence on reçoit 1530 kW/m² par an
On nous donne la consommation d’énergie pour 1
appartement : Eannuelle = 5.10 3 kWh
la surface de panneaux voltaïques sur le toit :
Spanneaux = 100 m²
le nombre d’appartements : 6
l’énergie lumineuse totale annuelle (Elum_tot_an) est :
Elum_tot_an = 1530 x Spanneaux
l’énergie électrique produite annuelle (Eélec_an) est :
Eélec_an = η x Elum_tot_an
On calcule donc la production annuelle des panneaux
solaires : Eélec_an on la divise par la consommation
énergétique annuelle d’un appartement: Eannuelle , on obtient
le nombre d’appartements nappartements que l’on peut
alimenter, On comparera donc avec le nombre
d’appartements de l’immeuble : 6
nappartements = Eélec_an / Eannuelle = η x Elum_tot_an / Eannuelle = η x 1530 x Spanneaux / Eannuelle = 0,8 x 1530 x 100 / 5.10 3 = 24,48

