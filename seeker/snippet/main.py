#date: 2022-09-09T17:14:04Z
#url: https://api.github.com/gists/6e3bb3c0c37851bafc541c1129dea0c1
#owner: https://api.github.com/users/sondhag

import random

Navn = input('Hva heter du, ærede lotterikommissær? Fyll inn under\n')
Vin = input('Ok, '+ str(Navn) + '. Hva heter vinen du har kjøpt?\n')
Beskrivelse_vin = input('Hvordan vil du beskrive vinen\n')
Deltakere = []
Deltakere_length = int(input('Hvor mange har vippset denne uken? Oppgi antall som nummer, ikke bokstaver\n'))
for idx in range(Deltakere_length):
    item = input('Oppgi et og et navn, til du har oppgitt alle ' + str(Deltakere_length) + '. Trykk enter etter hvert navn\n')
    Deltakere.append(item)
intro = 'Vinlotteri denne uken ved overlotterikommissær ' + Navn + '. Antall deltagere denne runder er '
Utro = 'Vinneren er '
Resultat = (random.choice(Deltakere))
klar = input('Er du klar til å kjøre trekningen?\n')
if klar == 'ja':
    print((intro) + str(len(Deltakere)))
    print((Utro) + (Resultat) + '!!!')
    print('Vinen er en ' + Vin + '. Den er ' + Beskrivelse_vin)
else:
    print('jaja, vi kjører uansett')
    print((intro) + str(len(Deltakere)))
    print((Utro) + (Resultat) + '!!!')
    print('Vinen er en ' + Vin + '. Den er ' + Beskrivelse_vin)
while True:
    promt=input('Er du fornøyd med programmet? ja/nei')
    if promt == 'nei':
        print('Jævla idiot, du stryke på eksamen. Prøv igjen')
    else:
      print('Bra, du forstår mange ting. Går sikker bra på eksamen')
      break
input('Håper du har vært en god, ærverdig lotterikommissær')