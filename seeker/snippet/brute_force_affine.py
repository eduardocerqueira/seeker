#date: 2024-09-13T16:44:57Z
#url: https://api.github.com/gists/4ab1e134faf18daa87f0d1451b236c8d
#owner: https://api.github.com/users/xHacka

def affine_decrypt(ciphertext, a, b):
    m = 256  # Byte range (0-255)
    a_inv = pow(a, -1, m)
    return bytes([(a_inv * (byte - b)) % m for byte in ciphertext])

def brute_force_affine(ciphertext):
    m = 256 # Max value 
    for a in range(1, m):
        try:
            for b in range(m):
                decrypted = affine_decrypt(ciphertext, a, b).decode("utf-8", errors="ignore")
                yield f'a={a}, b={b}: {decrypted}'
        except ValueError:
            continue

    return results


ciphertext = bytes.fromhex('9094939a8b8c8b8d868b97968c919088c0')
results = brute_force_affine(ciphertext)
for result in results:
    if result.isprintable() and result.isascii():
        print(result)
