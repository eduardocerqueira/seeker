#date: 2021-10-15T16:59:37Z
#url: https://api.github.com/gists/6226fa1b5de960497279b74e96d2e68c
#owner: https://api.github.com/users/paulnbrd

import time
import matplotlib.pyplot as plt


def diviseurs(n: int, positifs_uniquement: bool = False) -> list:
    """ Fonction pour récupérer tous les diviseurs d'un nombre """
    assert type(n) is int
    if positifs_uniquement:
        return [i for i in range(1, n + 1) if n % i == 0]
    return [i for i in range(-n, n + 1) if i != 0 and n % i == 0]


def diviseurs_stricts(n: int, positifs_uniquement: bool = False) -> list:
    """ Récupérer les diviseurs stricts d'un nombre entier """
    l = diviseurs(n, positifs_uniquement)
    l.remove(n)
    return l


def nombres_amiables(n1: int, n2: int) -> list:
    """ Tester si deux nombres sont amiables """
    diviseurs_n1 = diviseurs_stricts(n1, True)
    diviseurs_n2 = diviseurs_stricts(n2, True)

    sum_1 = sum(diviseurs_n1)
    sum_2 = sum(diviseurs_n2)
    return sum_1 == n2 and sum_2 == n1


def nombre_parfait(n: int) -> bool:
    """ Tester si un nombre est parfait """
    d = diviseurs_stricts(n, True)
    return sum(d) == n


def nombre_premier(n: int) -> bool:
    """ Tester si un nombre est premier """
    return diviseurs(n, True) == [1, n]


def crible_eratosthene(n: int) -> list:
    """ Fonction non optimisée pour un crible d'Ératosthène """
    l = [i for i in range(2, n + 1)]
    p = []

    while len(l) != 0:
        p.append(l[0])
        i = l[0]
        for k in l:
            if n % i != 0:
                del (l[l.index(k)])

    return p


def crible(n: int) -> list:
    """ Fonction optimisée pour un crible d'Ératosthène """
    liste_crible = [True for _ in range(2, n + 1)]
    index = 2

    while index <= (n + 1) / 2:
        for i in range(index * 2, n + 1, index):
            liste_crible[i - 2] = False
        index += 1
        while not liste_crible[index - 2]:
            index += 1

    return [i + 2 for i, e in enumerate(liste_crible) if e]


def mesure() -> None:
    """ Mesurer les performances des fonctions crible et _crible_eratosthene """
    vals = list(range(0, 2000, 850))

    crible_1_vals = []
    for v in vals:
        start = time.monotonic()
        crible_eratosthene(v)
        end = time.monotonic()
        crible_1_vals.append(end - start)

    crible_2_vals = []
    for v in vals:
        start = time.monotonic()
        crible(v)
        end = time.monotonic()
        crible_2_vals.append(end - start)

    plt.plot(vals, crible_1_vals, label="Fonction non optimisée")
    plt.plot(vals, crible_2_vals, label="Fonction optimisée")
    #plt.xlabel("Valeur de n")
    #plt.ylabel("Temps d'éxecution (en s)")
    #plt.legend(loc="upper left")
    plt.show()

def associes(n: int) -> list:
  r = []
  if nombre_premier(n) and False:
    return [(n, 1), (-n, -1)]
  
  for x in range(-n, n+1):
    for y in range(-n, n+1):
      if x*y == n and (y, x) not in r:
        r.append((x, y))
        break
  return r
        
      
def associesf(n: int) -> list:
  for i in associes(n):
    print(i)
