#date: 2022-12-13T16:58:33Z
#url: https://api.github.com/gists/489611af89b0443fd55028332f2dbc18
#owner: https://api.github.com/users/aldipratt

"""
Tugas 2:

Ubah Loops (print angka dalam range tertentu) pada gambar menjadi fungsi rekursif, dengan output yang sama.

clue: 
- 4-5 line aja
- Tidak perlu loops, hanya menggunakan rekursif.
- Menggunakan argument Positional, dan Kwargs.
- Print 1 biji doang d dlm fungsi.


Selamat mengerjakan wkwkwk
format kirim tugas sama sperti sebelumnya.
"""

def rekursif(awal , akhir) :
    if(awal <= akhir) :
        print(awal) 
        rekursif(awal + 1 , akhir)

rekursif(0,9)