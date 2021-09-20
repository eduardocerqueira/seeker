#date: 2021-09-20T17:10:19Z
#url: https://api.github.com/gists/db8532ac0a551f9ca02d8a23c3b24e1c
#owner: https://api.github.com/users/SWETRAK

class Zliczanie:
    ile = 0

    def __init__(self):
        Zliczanie.ile += 1
        self.nazwa = f'Obiekt o nazwie: {Zliczanie.ile}'



licznik_1 = Zliczanie()
licznik_2 = Zliczanie()
licznik_3 = Zliczanie()

print(f'Utworzono {Zliczanie.ile} obiektów klasy Zliczanie')
print(licznik_1.nazwa)
print(licznik_2.nazwa)
print(licznik_3.nazwa)