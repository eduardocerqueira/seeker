#date: 2023-03-31T17:01:04Z
#url: https://api.github.com/gists/7cfffd749aace7baffbdcead920dd5bc
#owner: https://api.github.com/users/suhedizayn

import random

kelimeler = ["elma", "armut", "portakal", "çilek", "kiraz", "karpuz"]
kelime = random.choice(kelimeler)

canlar = 6
tahmin_edilen = []
for i in range(len(kelime)):
    tahmin_edilen.append("_")

while canlar > 0 and "_" in tahmin_edilen:
    print(" ".join(tahmin_edilen))
    tahmin = input("Bir harf tahmin edin: ")

    if tahmin in kelime:
        for i in range(len(kelime)):
            if kelime[i] == tahmin:
                tahmin_edilen[i] = tahmin
    else:
        canlar -= 1
        print("Yanlış. Kalan can sayısı:", canlar)

if canlar == 0:
    print("Canların tükendi. Kelime: ", kelime)
else:
    print("Tebrikler! Kelimeyi buldun: ", kelime)
