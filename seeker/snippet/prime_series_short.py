#date: 2021-12-29T17:02:09Z
#url: https://api.github.com/gists/9fd52dad41d903afcc991eea5e07447c
#owner: https://api.github.com/users/galihboy

# https://blog.galih.eu
def deret_prima_v3(bmin, bmaks):
    bukan_prima = set([i for i in range(bmin, bmaks + 1) for j in range(2, i) if i % j == 0])
    return list(set(range(bmin, bmaks + 1)).symmetric_difference(bukan_prima))

print(deret_prima_v3(2, 10))
# output: [2, 3, 5, 7]
print(deret_prima_v3(200, 300))
# output: [257, 263, 269, 271, 277, 281, 283, 293, 211, 223, 227, 229, 233, 239, 241, 251]