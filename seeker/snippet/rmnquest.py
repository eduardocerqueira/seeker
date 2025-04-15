#date: 2025-04-15T16:32:06Z
#url: https://api.github.com/gists/e17172c3e611f5f217a1f9743c8ab566
#owner: https://api.github.com/users/techvolt-infos

Noyau observable :
Noyau observable si possede spin nucl diff
de 0 (I=/0). Cela implique un moment 
magne nucl qui peut interagir avec champ
magne externe
------------------------------
Principe spectro RMN :
Un spectromètre RMN utilise fort champ
magne statique (B0) pour aligner moments
magne des noyaux possedant un spin.
Une impulsion de radiofreq (RF) est envoyee
a freq de resonance des noyaux, provoquant
transition entre leurs etats denergie.
Apres cette excitation, les noyaux relaxent
en emettant un signal, appele FID 
(Free Induction Decay), qui, transformer 
via une FFT (transformee de Fourier) pour
obtenir un spectre. Ce spectre revele 
les environnements chimiques des noyaux 
observes
------------------------------
Anistrope solide - Isotrope liquide :
Dans liq, molecules bougent rapidement, 
ce qui moyenne interactions dans toutes
les direct, rendant deplacement chimique
isotrope. En revanche, dans solide, 
molecules sont figees dans des orientations
variees, deplacement chimique depend de 
lorientation du noyau par rapport a B0
------------------------------
Technique Anistrope => Isotrope:
Rotation à lAngle Magique (MAS, 
Magic Angle Spinning). 
Ech placer dans rotor incline à 54,7° 
par rapport au champ B0 et est
rapidement mis en rotation. 
Cette rotation moyenne les interactions 
anisotropes comme le deplacement chimique, 
donnant des signaux similaires à ceux 
d’un liq (spectre isotrope)
------------------------------
Bande rotation:
Artefact carac des spectres RMN du solide
obtenus sous MAS. Resulte de la rota 
partielle des interactions anisotropes 
et se traduit par des pics satellites 
espacer de la freq de rota autour du 
pic central
------------------------------
Interactions solide, influence B0:
-Déplacement chimique anisotrope - depend peu
de B0, fournit des infos sur l’env 
electronique.
-Interaction dipolaire - proportio a 1/r^3
(distance inter-nucl) et indep de B0.
Renseigne sur la geometrie locale.
-Interaction quadripolaire - depend de 
structure locale du champ electri, 
effet augmente avec B0 pour interactions
second ordre
-Couplage scalaire (J) - generalement petit, 
faiblement dépendant de B0

Ces interactions permettent dobtenir des
info structurales (positions, distances) et dynamiques 
(mouvements locaux)
------------------------------
Ref RMN elements :
-Proton: TMS (Tetramethylsilane) - utilise car 
chimiquement inerte et signal net.
-Carbone - idem que proton
-Silicium - TMS ou DSS (solution aqueuse),
pour sa stabilité chimique.
-Phosphore - Acide phosphorique (H3PO4), car est pur,
stable, et soluble dans l’eau
Ces ref fournissent une ech 
universelle de deplacement chimique
------------------------------
