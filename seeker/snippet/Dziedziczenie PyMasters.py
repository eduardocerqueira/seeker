#date: 2024-03-22T17:02:12Z
#url: https://api.github.com/gists/775f03807583f132dd7e30fd686e2d16
#owner: https://api.github.com/users/Pawelooo

class Pojazd:

    def __init__(self, predkosc: int, kolor: str) -> None:
        self.predkosc: int = predkosc
        self.kolor: str = kolor

    def jedz(self) -> None:
        print('Pojazd jedzie.')

    def zatrzymaj(self) -> None:
        print("Pojazd zatrzymuje się.")


class Autobus(Pojazd):

    def __init__(self, predkosc: int, kolor: str) -> None:
        super().__init__(predkosc, kolor)
        self.limit_miejsc = 50

    def wpusc_osobe(self) -> None:
        self.limit_miejsc -= 1

    def wypusc_osobe(self) -> None:
        self.limit_miejsc += 1


class Czolg(Pojazd):

    def __init__(self, predkosc: int, kolor: str, ilosc_naboi: int) -> None:
        super().__init__(predkosc, kolor)
        self.ilosc_naboi = ilosc_naboi

    def wystrzal(self) -> None:
        print('    ==>     X')

    def otworz_klape(self) -> None:
        print('Otwiera klape')
