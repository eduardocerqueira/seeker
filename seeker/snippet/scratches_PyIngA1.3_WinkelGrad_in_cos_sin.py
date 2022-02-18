#date: 2022-02-18T16:54:01Z
#url: https://api.github.com/gists/31c1c293d6ceb4a0cabfce5ca942b597
#owner: https://api.github.com/users/HelloRamo

from math import *
print("Winkel Bogenmaß (bis 360°) in cos/sin")
print("-----------------------------------------------------")
print()

winkel_grad = float(input("Bitte einen Winkel in Grad (max. 360°) eingeben: "))
winkel_rad = radians(winkel_grad)
sinus = sin(winkel_rad)
cosinus = cos(winkel_rad)

print()
print("Winkel:",winkel_grad, " Sinuswert: ",sinus," Kosinuswert: ",cosinus)


