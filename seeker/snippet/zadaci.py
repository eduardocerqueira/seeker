#date: 2023-04-05T16:59:12Z
#url: https://api.github.com/gists/eef692d2bc42e005773a71d82716bf6e
#owner: https://api.github.com/users/ac0x

#25
'''
s = "+23-2-32+4-22-4"
brojac = 0
for i in range(len(s)):
    if s[i] == "-" and i + 1 < len(s):
        if i + 2 < len(s) and s[i+1].isdigit() and not s[i+2].isdigit():
            brojac+=1
        elif i + 2 == len(s) and s[i+1].isdigit():
            brojac+=1

print(brojac)

s = "+32-1+23-23"
brojac = 0

for i in range(len(s)):
    if s[i] =="-":
        if i + 2 < len(s) and s[i+1].isdigit() and not s[i+2].isdigit():
            brojac+=1
        if i + 2 == len(s) and s[i+1].isdigit():
            brojac+=1

print(brojac)'''

#21 Prosjecna ocjena svih ucenika
'''
n = int(input("Unesi broj ucenika: "))
k = int(input("Unesi broj ucenika na drugoj str: "))
p1 = float(input("Unesi prosjecnu ocjenu za prvu stranu: "))
p2 = float(input("Unesi prosjecnu ocjenu za drugu stranu: "))

#Ulaz 80, 30, 78.2 , 89.3

ucenika_p1 = n - k

ukupan_br_poena = float((ucenika_p1 * p1) + (k * p2))
prosjecan_br_poena = ukupan_br_poena / n
print(prosjecan_br_poena)'''

#22 da li je pero ubrao vise jabuka
'''
p = int(input("Unesi broj jabuka koje je Petar ubrao: "))
m = int(input("Unesi broj jabuka koje je Milos ubrao: "))
uspio = False

if p <= m:
    uspio = False
else:
    uspio = True

print(uspio)'''

#23 da li je mail stigao u toku radnog vremena
'''
vrijeme_maila = int(input("Unesite vrijeme kada je mail stigao: "))

if vrijeme_maila >= 9 and vrijeme_maila <= 17:
    print("Mail je poslat u toku radnog vremena")
else:
    print("Mail nije poslat u toku radnog vremena")'''

#24  buka
'''
hour = int(input("Unesi sate: "))

if hour <= 6 or (hour >= 13 and hour <= 17) or hour >= 22:
    print("Ne smiju da se izvode radovi ")
else:
    print("Smiju da se izvode radovi")'''

#25
'''
a = float(input("Unesi stranicu a: "))
b = float(input("Unesi stranicu b: "))
c = float(input("Unesi stranicu c: "))
#da li se moze napraviti basta u obliku trougla

if a + b > c and a + c > b and c + b > a:
    print("Moze se napraviti basta")
else:
    print("Ne moze se napraviti basta")'''

#26
'''
x = int(input("unesi broj: "))

if x % 2 == 0:
    print("broj je paran")
else:
    print("Nije paran")'''

#27
'''
cijena_hleba = 0.8
posklj = 0.8 * 1.1
pojeft = posklj * 0.9

print("Nova cijena hleba je: " + str(pojeft))'''

#28
'''
prva = 4
druga = 3
treca = 3

najmanja = 0
if najmanja > prva:
    najmanja = prva
if najmanja > druga:
    najmanja = druga
elif najmanja > treca:
    najmanja = treca

if najmanja != prva:
    print(prva)
if najmanja != druga:
    print(druga)
if najmanja != treca:
    print(treca)'''

#29

x = "321"
y = "123"

k = int(input("Unesi indkes prvog broja: "))
l = int(input("Unesi indeks drugog broja: "))

result = int(x.find(k)) + int(y.find(l))
print(result)






















