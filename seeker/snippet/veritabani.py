#date: 2023-11-08T16:43:10Z
#url: https://api.github.com/gists/964c5a9ea2d2eb65209c2a46d723f9e2
#owner: https://api.github.com/users/civanpython

import tkinter as tk
import json
import turtle

# JSON dosyasından öğrenci verilerini yükle
def yukle_json():
    try:
        with open("ogrenci_verileri.json", "r") as json_file:
            ogrenci_verileri = json.load(json_file)
            return ogrenci_verileri
    except FileNotFoundError:
        return {}

# GUI penceresini oluşturun
root = tk.Tk()
root.title("Öğrenci Bilgi Saklayıcı")

# Öğrenci verilerini yükle
ogrenci_verileri = yukle_json()

# Ekleme işlevi
def ogrenci_ekle():
    ad = entry_ad.get()
    soyad = entry_soyad.get()
    numara = entry_numara.get()
    
    ogrenci_verileri[numara] = {"Ad": ad, "Soyad": soyad}
    
    # Veriyi JSON dosyasına kaydet
    with open("ogrenci_verileri.json", "w") as json_file:
        json.dump(ogrenci_verileri, json_file)
    
    # Veriyi göster
    veriyi_goster()

# Veriyi gösterme işlevi
def veriyi_goster():
    # Turtle ile öğrenci bilgisini göster
    turtle.clear()
    turtle.penup()
    y = 100
    for numara, bilgi in ogrenci_verileri.items():
        bilgi_metni = f"Numara: {numara}, Ad: {bilgi['Ad']}, Soyad: {bilgi['Soyad']}"
        turtle.goto(0, y)
        turtle.write(bilgi_metni, align="center", font=("Arial", 12, "normal"))
        y -= 30

# GUI bileşenlerini oluşturun
label_ad = tk.Label(root, text="Ad:")
entry_ad = tk.Entry(root)
label_soyad = tk.Label(root, text="Soyad:")
entry_soyad = tk.Entry(root)
label_numara = tk.Label(root, text="Numara:")
entry_numara = tk.Entry(root)
ekle_button = tk.Button(root, text="Öğrenci Ekle", command=ogrenci_ekle)

# JSON verilerini gösterme düğmesi
goster_button = tk.Button(root, text="JSON Verileri Göster", command=veriyi_goster)

# GUI bileşenlerini yerleştirin
label_ad.pack()
entry_ad.pack()
label_soyad.pack()
entry_soyad.pack()
label_numara.pack()
entry_numara.pack()
ekle_button.pack()
goster_button.pack()

# Turtle grafik penceresini ayarlayın
turtle.setup(400, 400)
turtle.hideturtle()

# GUI penceresini başlatın
root.mainloop()