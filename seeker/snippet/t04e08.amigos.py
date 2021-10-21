#date: 2021-10-21T16:55:39Z
#url: https://api.github.com/gists/6d9359a8c5967e89b6c6d45f24cc44e2
#owner: https://api.github.com/users/juanfal

# t04e08.amigos.py
# juanfc 2021-10-21
# https://gist.github.com/6d9359a8c5967e89b6c6d45f24cc44e2

print("Entrar dos números enteros")
n = int(input("¿n: "))
m = int(input("¿m: "))


sn = 0
d = 1
while d < n:
    if n % d == 0:
        sn += d
    d += 1

sm = 0
d = 1
while d < m:
    if m % d == 0:
        sm += d
    d += 1


if sn == m and sm == n:
    print(f"Los número {n} y {m} son amigos")
else:
    print(f"Los número {n} y {m} NO son amigos")

