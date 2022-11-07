#date: 2022-11-07T17:16:46Z
#url: https://api.github.com/gists/fdfec7a3d36ea9de6d76be74f698d01b
#owner: https://api.github.com/users/rguiguet-teledyne

__version__ = "1.0.0"
__author__ = "RGU"

def is_prime(nb: int) -> bool:
    if nb < 2:
        result = False
    else:
        result = True
        for div in range(2, nb):
            if nb % div == 0:
                result = False
                break
    return result

def add(x: float, y: float = 0) -> float:
    return x+y

def add2(*x):
    sum = 0
    for i in x:
        sum += i
    return sum