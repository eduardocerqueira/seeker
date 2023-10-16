#date: 2023-10-16T17:06:10Z
#url: https://api.github.com/gists/a7682e8eb9d7f803cc68cdecaf07f72b
#owner: https://api.github.com/users/cielhaidir

import requests

endpoint = 'http://akses.poliupg.ac.id/api/dosenpegawai'
nip = '123'
api_key = 'ulPuKei4wYs68nolVWMBKyVboWrrmB7b'

# Membangun URL dengan parameter nip
url = f'{endpoint}?nip={nip}'

# Membangun header
headers = {
    'key': api_key
}

# Membuat permintaan GET
response = requests.get(url, headers=headers)

# Memeriksa jika permintaan berhasil
if response.status_code == 200:
    data = response.json()
    # Menggunakan data yang diterima dari API
    print(data)
else:
    print(f'Error: {response.status_code} - {response.text}')
