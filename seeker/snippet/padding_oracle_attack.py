#date: 2024-11-15T17:11:52Z
#url: https://api.github.com/gists/9363467f6a04fb4f417b3ec68e208b31
#owner: https://api.github.com/users/virgilhem

import argparse
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


def debug_print(level, *args, **kwargs):
    if debug_level >= level:
        print(*args, **kwargs)


def pad_pkcs7(data):
    padding_len = 16 - len(data) % 16
    return data + bytes([padding_len] * padding_len)


def unpad_pkcs7(data):
    padding_len = data[-1]
    if data[-padding_len:] != bytes([padding_len] * padding_len):
        raise ValueError("Padding incorrect")
    return data[:-padding_len]


def encrypt(plaintext):
    key = get_random_bytes(16)
    iv = get_random_bytes(16)

    def vuln_decrypt(ciphertext):
        cipher = AES.new(key, AES.MODE_CBC, iv)
        try:
            plaintext = cipher.decrypt(ciphertext)
            unpad_pkcs7(plaintext)
            return True
        except ValueError:
            return False

    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_plaintext = pad_pkcs7(plaintext)
    ciphertext = iv + cipher.encrypt(padded_plaintext)

    debug_print(0, f"key={key.hex()}\niv={iv.hex()}\nciphertext={ciphertext.hex()}\n")
    return ciphertext, vuln_decrypt


def padding_oracle_attack(ciphertext, oracle):
    plaintext = b""
    cipher_blocks = [ciphertext[i:i + 16] for i in range(0, len(ciphertext), 16)]

    for i in range(len(cipher_blocks) - 1, 0, -1):
        prev_cipher_block = bytearray(cipher_blocks[i - 1])
        target_block = cipher_blocks[i]
        decrypted_block = bytearray(16)
        count = 0

        for pad in range(1, 16 + 1):

            # ajuste le bloc C'_{i-1} avec les octets j déjà découverts
            for j in range(1, pad):
                # pour tout j tel que P_i[-j] = D_K(C_i)[-j] ^ C_{i-1}[-j] est connu
                # le padding cible pad est obtenu ssi D_K(C_i)[-j] ^ C'_{i-1}[-j] = pad
                # on doit donc avoir: P_i[-j] = pad ^ C'_{i-1}[-j] ^ C_{i-1}[-j]
                prev_cipher_block[-j] = pad ^ decrypted_block[-j] ^ cipher_blocks[i - 1][-j]
                debug_print(2, f"C'_{i-1}[{16-j}] <- {pad} ^ P_{i}[{16-j}] ^ C_{i-1}[{16-j}]")

            for k in range(256):
                count += 1
                prev_cipher_block[-pad] = k
                forged_ciphertext = bytes(prev_cipher_block) + target_block

                # soumet C'_{i-1}||C_i avec C'_{i-1}[-pad] = k
                if oracle(forged_ciphertext):
                    # D_K(C_i)[-pad] ^ C'_{i-1}[-pad] = pad, or C_i = E_K(P_i ^ C_{i-1})
                    # on a donc l'égalité: P_i[-pad] ^ C_{i-1}[-pad] ^ C'_{i-1}[-pad] = pad
                    # d'où la valeur de P_i[-pad]
                    decrypted_byte = k ^ pad ^ cipher_blocks[i - 1][-pad]
                    debug_print(2, f"P_{i}[{16-pad}] <- {k} ^ {pad} ^ C_{i-1}[{16-pad}]")

                    # élimine les faux positifs liés aux octets de padding
                    if decrypted_byte == pad and pad != 16:
                        prev_cipher_block[-pad-1] ^= 1
                        if not oracle(bytes(prev_cipher_block) + target_block):
                            prev_cipher_block[-pad-1] ^= 1
                            debug_print(2, f"{pad} not a padding byte for block {i}")
                            continue

                    decrypted_block[-pad] = decrypted_byte
                    debug_print(1, f"P_{i}[{16-pad}] <- {decrypted_byte} [k={k}]")
                    break

        plaintext = bytes(decrypted_block) + plaintext
        debug_print(0, f"P_{i} <- {decrypted_block} [{count} iterations]")

    return plaintext


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="count", default=0)
parser.add_argument("plaintext", nargs= "**********"="This is a secret message!")

args = parser.parse_args()
plaintext = args.plaintext.encode()
debug_level = int(args.debug)

ciphertext, oracle = encrypt(plaintext)
recovered_plaintext = padding_oracle_attack(ciphertext, oracle)
recovered_plaintext = unpad_pkcs7(recovered_plaintext)

print(recovered_plaintext.decode('utf-8'))