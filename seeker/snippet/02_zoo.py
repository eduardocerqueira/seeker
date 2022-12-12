#date: 2022-12-12T16:57:44Z
#url: https://api.github.com/gists/259ce7e032e3ca6c89d5fab9d41bdedb
#owner: https://api.github.com/users/brabemi

class Zviratko:
    def __init__(self, jmeno, pocet_nohou=4, hladove=False):
        self.jmeno = jmeno.capitalize()
        self.pocet_nohou = pocet_nohou
        self.zavazane_tkanicky = True
        self.hladove = hladove

    def snez(self, jidlo):
        print("{}: {} mi chutná!".format(self.jmeno, jidlo))

    def udelej_zvuk(self):
        print("{}: {}!".format(self.jmeno, self.zvuk))


class Jehnatko(Zviratko):
    # class proměnná - je společná pro celou třídu
    zvuk = 'Bééé'

# class Kotatko(Zviratko):
#     def __init__(self, jmeno, pocet_nohou=4, hladove=False):
#         super().__init__(jmeno, pocet_nohou, hladove)
#         self.zvuk = 'Mňau'

#     def zamnoukej(self):
#         print("{}: Mňau!".format(self.jmeno))

class Kotatko(Zviratko):
    def udelej_zvuk(self):
        print("{}: Mňau!".format(self.jmeno))

class Stenatko(Zviratko):
    # def zastekej(self):
    def udelej_zvuk(self):
        print("{}: Haf!".format(self.jmeno))

class Kuzlatko(Zviratko):
    # def zamec(self):
    def udelej_zvuk(self):
        print(f'{self.jmeno}: Méééé!')

# https://pastebin.com/CS3h8cLJ
# vytvořte novou třídu Telatko
# Telatko dedí z Zviratko
# Telatko má metodu zabuc()
class Telatko(Zviratko):
    # def zabuc(self):
    def udelej_zvuk(self):
        print(f'{self.jmeno}: Bůůůů!')

class Papousce(Zviratko):
    # def __init__(self, jmeno, pocet_nohou=2):
    #     self.jmeno = jmeno.capitalize()
    #     self.pocet_nohou = pocet_nohou

    def __init__(self, jmeno, pocet_nohou=2, hladove=False):
        super().__init__(jmeno, pocet_nohou)
        self.zavazane_tkanicky = False

    # def rekni_neco(self):
    def udelej_zvuk(self):
        print(f'{self.jmeno}: Muž přes palubu!')

# vyrob třídu Hadatko
# hadatko má 0 nohou
# had udela z s ve jmenu sss
# had Severus -> Ssseverusss
# had má metodu zasyc

class Hadatko(Zviratko):
    def __init__(self, jmeno, pocet_nohou=0, hladove=False):
        jmeno = jmeno.lower().replace('s', 'sss')
        super().__init__(jmeno, pocet_nohou, hladove)
        self.zvuk = 'sss'
        # self.jmeno = self.jmeno.replace('s', 'sss')
        # self.jmeno = self.jmeno.replace('S', 'Sss')

    # def zasyc(self):
    def udelej_zvuk(self):
        print(f'{self.jmeno}: Sssss!')


mourek = Kotatko('Mourek', hladove=True)
alik = Stenatko('Alík')
liza = Kuzlatko('Líza', hladove=True)
milka = Telatko('Milka')
pepik = Papousce('pepík', hladove=True)
severus = Hadatko('Severus')
oskar = Jehnatko('Oskar')

zoo = [
    mourek,
    alik,
    liza,
    milka,
    pepik,
    severus,
    oskar
]

for zvire in zoo:
    zvire.udelej_zvuk()
    zvire.snez('granule')
    print(zvire.jmeno, zvire.pocet_nohou, zvire.zavazane_tkanicky, zvire.hladove)
    # ošklivá varianta jak dělat zvuk zvířete
    # if isinstance(zvire, Kotatko):
    #     zvire.zamnoukej()
    # elif isinstance(zvire, Stenatko):
    #     zvire.zastekej()
