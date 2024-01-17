#date: 2024-01-17T16:56:03Z
#url: https://api.github.com/gists/0caf81b0abb6267f9c1a5e4b6caf7b0a
#owner: https://api.github.com/users/orjanv

from time import sleep
print("Vi prøver Kaprekar's routine\n")

a = input("Velg et tre eller firesifret tall, med minst to ulike siffer: ")
nummer = "0"
desc_a = "".join(reversed(sorted(list(a))))
asc_a = "".join(sorted(list(a)))
steg = 0
siffer = len(a)

print(f"\nVi finner størst {desc_a} og minst tall {asc_a} fra sifrene og begynner rutinen...\n")
while nummer not in ("6174", "495"):
    nummer = str(int(desc_a) - int(asc_a))
    nummer = nummer.zfill(siffer)
    print(f"{desc_a} - {asc_a} = {nummer}")
    desc_a = "".join(reversed(sorted(list(nummer))))
    asc_a = "".join(sorted(list(nummer)))
    sleep(0.5)
    steg += 1

print(f"\nVi har kommet frem til Kaprekar's: {nummer} etter {steg} steg")

