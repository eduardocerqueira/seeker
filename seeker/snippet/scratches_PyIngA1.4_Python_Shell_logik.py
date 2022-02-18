#date: 2022-02-18T16:54:01Z
#url: https://api.github.com/gists/31c1c293d6ceb4a0cabfce5ca942b597
#owner: https://api.github.com/users/HelloRamo

from math import *
print("Logik um auf wahr/falsch zu prüfen")
print("-----------------------------------------------------")
print()
# verschiedene Ansätze zB per Logik.Zeichen
# ((8 < 5) or (4 != 7)) or not (not (4 == 5))

# per Variable endet mit true (besser)
ergebnis = ((8 < 5) or (4 != 7)) or not (not (4 == 5))
print(ergebnis)