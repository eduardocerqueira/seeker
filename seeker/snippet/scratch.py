#date: 2022-07-01T17:00:18Z
#url: https://api.github.com/gists/4015702d1d26d887dc2000ca6ed48c03
#owner: https://api.github.com/users/sl0wlearn3r

import json
import random

import requests
import bs4

url = 'https://sozluk.gov.tr/icerik'
response = requests.get(url, headers={
    "Host": "sozluk.gov.tr",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Prefer": "safe",
    "X-Requested-With": "XMLHttpRequest",
    "Connection": "keep-alive",
    "Referer": "https://sozluk.gov.tr/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Sec-GPC": "1",
})


j = json.loads(response.content.decode())

kelime_kendi = j['kelime'][0]['madde']
kelime_anlam = j['kelime'][0]['anlam']
print(kelime_kendi, ":",kelime_anlam)

kelime_analiz = kelime_anlam.split()
print(kelime_analiz)



iterate_list =[1]

randomnumber = random.randint(1,10)

arbitrary_pair = 15

while arbitrary_pair == 15:



    for i in iterate_list:
        sorgulama = input("Kelimenin anlamadığınız bir kısmı var mı?")

        if sorgulama == "hayir":
            arbitrary_pair = 16
            break
        else:
            iterate_list.append(randomnumber)
            #print(iterate_list)

        new_url = "https://sozluk.gov.tr/gts?ara="

        addition_sign = "+"
        sorgulanan_kelimeler = sorgulama.split()

        if len(sorgulanan_kelimeler) >= 2:
            for e in sorgulanan_kelimeler:
                new_url = new_url + addition_sign + e
        else:
            new_url = new_url + sorgulanan_kelimeler[0]
        #print(new_url)
        response = requests.get(new_url, headers={
            "Host": "sozluk.gov.tr",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0",
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Prefer": "safe",
            "X-Requested-With": "XMLHttpRequest",
            "Connection": "keep-alive",
            "Referer": "https://sozluk.gov.tr/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Sec-GPC": "1",

        })
        j = json.loads(response.content.decode())
        bir_sonraki_kelime = j
        #print ("jnin sonuclari:",j)
        print( " \n \n" , j[0]['anlamlarListe'][0]['anlam'])



else:
    exit_warning = input("Gercekten uygulamadan cikmak mi istiyorsunuz?")
    if exit_warning == 'hayir' or exit_warning =='yok':
        sys.exit







'''
GET /icerik HTTP/1.1
Host: sozluk.gov.tr
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0
Accept: application/json, text/javascript, */*; q=0.01
Accept-Language: en-US,en;q=0.5
Accept-Encoding: gzip, deflate, br
Prefer: safe
X-Requested-With: XMLHttpRequest
Connection: keep-alive
Referer: https://sozluk.gov.tr/
Cookie: _ga=GA1.3.1623371165.1601284120
Sec-Fetch-Dest: empty
Sec-Fetch-Mode: cors
Sec-Fetch-Site: same-origin
Sec-GPC: 1
'''