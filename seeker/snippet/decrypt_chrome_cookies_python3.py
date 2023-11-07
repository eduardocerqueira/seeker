#date: 2023-11-07T17:02:08Z
#url: https://api.github.com/gists/a5e814ffe9369e47bf1e633456da2a34
#owner: https://api.github.com/users/jackson5sec

import sqlite3
import sys
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import binascii
import json
# Python3 script
def decrypt_payload(cipher, payload):
    return cipher.decrypt(payload)

def generate_cipher(aes_key, iv):
    return AES.new(aes_key, AES.MODE_GCM, iv)

 "**********"d "**********"e "**********"f "**********"  "**********"d "**********"e "**********"c "**********"r "**********"y "**********"p "**********"t "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"( "**********"c "**********"i "**********"p "**********"h "**********"e "**********"r "**********"t "**********"e "**********"x "**********"t "**********", "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"_ "**********"k "**********"e "**********"y "**********") "**********": "**********"
    try:
        initialisation_vector = ciphertext[3:15]
        encrypted_password = ciphertext[15: "**********"
        cipher = "**********"
        decrypted_pass = "**********"
        decrypted_pass = decrypted_pass.decode('utf-8')
        return decrypted_pass
    except Exception as e:
        print(f"Error: {str(e)}")
        print("[ERR] Unable to decrypt. Please check.")
        return ""

if len(sys.argv) != 3:
    print("Usage: python script.py <cookies_file_path> <static_aes_key_hex>")
    sys.exit(1)

cookies_file_path = sys.argv[1]
static_aes_key_hex = sys.argv[2]

try:
    static_aes_key = binascii.unhexlify(static_aes_key_hex)
except binascii.Error:
    print("Error: Invalid hexadecimal key.")
    sys.exit(1)

conn = sqlite3.connect(cookies_file_path)
cursor = conn.cursor()

cursor.execute('SELECT host_key, expires_utc, name, path, encrypted_value FROM cookies')
cookies = []

for row in cursor.fetchall():
    try:
        host_key, expires_utc, name, path, encrypted_value = row
        decrypted_value = "**********"
        cookie = {
            "domain": host_key,
            "expirationDate": expires_utc,
            "name": name,
            "path": path,
            "value": decrypted_value
        }
        cookies.append(cookie)
    except Exception as e:
        print(f"Error decrypting cookie: {str(e)}")

conn.close()

# Print cookies in JSON format
print(json.dumps(cookies, indent=4))