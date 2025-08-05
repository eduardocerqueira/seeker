#date: 2025-08-05T16:43:14Z
#url: https://api.github.com/gists/45c3be1cd51266854ecfb02634861f6b
#owner: https://api.github.com/users/halfe-dev

import tkinter as tk
import tkinter.font as tkFont

alfabe = "abcdefghijklmnopqrstuvwxyzäöüß"

def bfcoz(mesaj, max_q=1):
    olasi_sonuclar = []
    for k in range(1, len(alfabe)):
        cozulmus = ""
        for i in mesaj:
            if i == "q":
                cozulmus += " "
            elif i.lower() in alfabe:
                konum = alfabe.index(i.lower())
                konum -= k
                yeni_harf = alfabe[konum % len(alfabe)]
                cozulmus += yeni_harf.upper() if i.isupper() else yeni_harf
            else:
                cozulmus += i
        q_sayisi = cozulmus.count("q")
        if q_sayisi <= max_q:
            olasi_sonuclar.append((k, cozulmus, q_sayisi))

    olasi_sonuclar.sort(key=lambda x: x[2])
    return olasi_sonuclar


def sifrele():
    mesaj = giris_kutusu.get().strip().lower()
    try:
        k = int(sayi_kutusu.get())
    except ValueError:
        cikti_kutusu.config(text="Geheimzahl muss ein Zahl sein!")
        return

    sifreli_mesaj = ""

    for i in mesaj:
        if i.isspace():
            sifreli_mesaj += "q"
        elif i in alfabe:
            konum = alfabe.index(i)
            konum += k
            yeni_harf = alfabe[konum % len(alfabe)]
            sifreli_mesaj += yeni_harf
        else:
            sifreli_mesaj += i
    cikti_kutusu.config(text=sifreli_mesaj)

def coz():
    mesaj = giris_kutusu.get().strip().lower()
    try:
        k = int(sayi_kutusu.get())
    except ValueError:
        cikti_kutusu.config(text="Geheimzahl muss ein Zahl sein!")
        return

    cozulmus_mesaj = ""

    for i in mesaj:
        if i == "q":
            cozulmus_mesaj += " "
        elif i in alfabe:
            konum = alfabe.index(i)
            konum -= k
            yeni_harf = alfabe[konum % len(alfabe)]
            cozulmus_mesaj += yeni_harf
        else:
            cozulmus_mesaj += i
    cikti_kutusu.config(text=cozulmus_mesaj)

def otomatik_coz():
    mesaj = giris_kutusu.get().strip()
    cozumler = bfcoz(mesaj, max_q=5)

    if not cozumler:
        cikti_kutusu.config(text="Keine sinnvolle Lösung gefunden.")
        return

    liste_kutusu.delete(0, tk.END)
    for k, sonuc, q_sayisi in cozumler[:10]:
        liste_kutusu.insert(tk.END, f"Geheimzahl {k}: {sonuc}")

    metin = "Mögliche Lösungen:\n"
    for k, sonuc, q_sayisi in cozumler[:5]:
        metin += f"Geheimzahl {k}: {sonuc} (q: {q_sayisi})\n"

    cikti_kutusu.config(text=metin)

def secimi_goster(evt):
    secilen_index = liste_kutusu.curselection()
    if not secilen_index:
        return
    secilen = liste_kutusu.get(secilen_index)
    cikti_kutusu.config(text=secilen)

def kopyala():
    metin = cikti_kutusu.cget("text")
    if metin:
        pencere.clipboard_clear()
        pencere.clipboard_append(metin)
        pencere.update()


pencere = tk.Tk()
font_boyut = tkFont.Font(family="Helvetica", size=14)
pencere.geometry("600x500")
pencere.title("Nachricht Versclüsseln / Entschlüsseln")

tk.Label(pencere, text="Nachricht:", font=font_boyut).pack()
giris_kutusu = tk.Entry(pencere, width=50, font=font_boyut)
giris_kutusu.pack()

tk.Label(pencere, text="Geheimzahl:", font=font_boyut).pack()
sayi_kutusu = tk.Entry(pencere, font=font_boyut)
sayi_kutusu.pack()

tk.Button(pencere, text="Verschlüsseln", command=sifrele, font=font_boyut, width=20, height=2).pack()
tk.Button(pencere, text="Entschlüsseln", command=coz, font=font_boyut, width=20, height=2).pack()
tk.Button(pencere, text="Automatisch Entschlüsseln", command=otomatik_coz, font=font_boyut, width=20, height=2).pack()
tk.Button(pencere, text="Ergebnis kopieren", command=kopyala, font=font_boyut, width=20, height=2).pack(pady=5)


liste_kutusu = tk.Listbox(pencere, width=60, height=6, font=font_boyut)
liste_kutusu.pack(pady=5)
liste_kutusu.bind("<<ListboxSelect>>", secimi_goster)

cikti_kutusu = tk.Label(pencere, text="", wraplength=400, font=font_boyut)
cikti_kutusu.pack()

pencere.mainloop()