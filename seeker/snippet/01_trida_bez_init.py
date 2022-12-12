#date: 2022-12-12T16:57:44Z
#url: https://api.github.com/gists/259ce7e032e3ca6c89d5fab9d41bdedb
#owner: https://api.github.com/users/brabemi

class Kotatko:
    def __init__(self, jmeno, pocet_nohou=4, jidlo=''):
        self.jmeno = jmeno
        self.pocet_nohou = pocet_nohou
        if jidlo != '':
            self.vybirave = True
            self.oblibene_jidlo = jidlo
        else:
            self.vybirave = False

    def __repr__(self):
        return 'ahoj'

    def _moje_metoda(self):
        pass

    def snez(self, jidlo):
        if self.vybirave:
            if jidlo == self.oblibene_jidlo:
                print("{}: {} mi chutná!".format(self.jmeno, jidlo))
            else:
                print("{}: {} mi nechutná!".format(self.jmeno, jidlo))
        else:
            print("{}: {} mi chutná!".format(self.jmeno, jidlo))


    def zamnoukej(self):
        print("{}: Mňau!".format(self.jmeno))

# pokud by třída neměla __init__ tak bych musel volat funkci, která by mi správně vyrobila nový objekt
# to je nešikovné, tak je lepší dát podobný kód rovnou do initu
# def vyrob_kotatko(jmeno):
#     mourek = Kotatko()
#     print(mourek)
#     mourek.pojmenuj('Mourek')
#     # mourek.jmeno = 'Mourek'
#     return mourek

# mourek = vyrob_kotatko('Mourek')

mourek = Kotatko('Mourek', jidlo='ryba')
print(mourek)
# mourek.snez('granule')
# mourek.zamnoukej()
