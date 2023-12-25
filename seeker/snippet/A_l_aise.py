#date: 2023-12-25T16:58:36Z
#url: https://api.github.com/gists/3900243411a2d34b1465eee16d39309e
#owner: https://api.github.com/users/morphisme

# cf https://hackropole.fr/fr/challenges/crypto/fcsc2022-crypto-a-laise/
# À l'aise
# intro crypto FCSC 2022

ct = "Gqfltwj emgj clgfv ! Aqltj rjqhjsksg ekxuaqs, ua xtwk n'feuguvwb gkwp xwj, ujts f'npxkqvjgw nw tjuwcz ugwygjtfkf qz uw efezg sqk gspwonu. Jgsfwb-aqmu f Pspygk nj 29 cntnn hqzt dg igtwy fw xtvjg rkkunqf."
key = 'FCSC'

alphabet = [chr(ord('A')+k) for k in range(26)]
table = [alphabet]
for k in range(1, 26):
    table.append(table[k-1][1:] + [table[k-1][0]])

pt = ''
k = 0
for c in ct:
    if 'A' <= c and c <= 'Z':
        base = ord('A')
    elif 'a' <= c and c <= 'z':
        base = ord('a')
    else:
        pt += c
        continue
    i = ord(key[k]) - ord('A')
    k = (k+1) % len(key)
    j = table[i].index(chr(ord('A') + ord(c) - base))
    pt += chr(base + j)
print(pt)
