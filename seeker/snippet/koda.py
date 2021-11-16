#date: 2021-11-16T16:55:16Z
#url: https://api.github.com/gists/4db8b2eb570d22f50fa19cd076d4155d
#owner: https://api.github.com/users/jO-Osko

import requests
import re
import threading
import time
import random

import queue

STEVILO_STRANI = 50

STEVILO_THREADOV = 7

za_obdelavo = queue.Queue()

trajanja = []

for j in range(STEVILO_STRANI):
    trajanja.append(j)

random.shuffle(trajanja)

for tr in trajanja:
    za_obdelavo.put_nowait(tr)


def predeluj(id_taska):
    while True:
        if not za_obdelavo.empty():
            
            stran = za_obdelavo.get_nowait()
            
            time.sleep(stran / 5)
            
            print("TASK:", id_taska, "OBDELUJE STRAN:", stran)
        else:
            break
    print("TASK", id_taska, "KONČANO")
        
niti = []
for j in range(STEVILO_THREADOV):
    nit = threading.Thread(
        target=predeluj,
        args=[ j ]
    )
    nit.start()
    niti.append(nit)

print("VSI SO ZAČELI")

for nit in niti:
    nit.join()

print("VSI SO KONČALI")




# def get_url(url):
#     page = requests.get(url)

#     return page.text



# def dobi_oglase(webpage):
#     pattern = r'<li class="EntityList-item EntityList-item--Regular(.*?)</article>'
#     regexp = re.compile(pattern, re.DOTALL)

#     return re.findall(regexp, webpage)
#     return re.findall(pattern, webpage, flags=re.DOTALL)


# vsi_oglasi = []

# def dobi_oglase_iz_strani(st_strani):

#     spletna_stran = get_url("https://www.bolha.com/avto-oglasi?page=" + str(st_strani))
#     oglasi = dobi_oglase(spletna_stran)
    
#     vsi_oglasi.extend(oglasi)

# vse_niti = []
# tt = time.time()
# for j in range(1,2):
#     nit = threading.Thread(
#         target=dobi_oglase_iz_strani,
#         args=(j,)
#     )
#     nit.start()
#     # nit.join()
#     vse_niti.append(nit)

# print("Odposlal vse, sedaj čakam")
# for t in vse_niti:
#     t.join()
# print("Vsi so končali", len(vsi_oglasi))
# print()
# #print(vsi_oglasi)
# print(time.time() - tt)



# # for o in oglasi:
# #     pattern_naslov = r'<h3.*?><a.*?>(?P<naslov>.*?)</a></h3>'
# #     pattern_rabljenost = r'<div class="entity-description-main">\s*(?P<rabljenost>.*?)'
# #     pattern = (
# #         pattern_naslov + r'.*' + 
# #         pattern_rabljenost + 
# #         r'<br />\s*(?P<leto>.*?)<br />.*?</span>(?P<lokacija>.*?)<br />'
# #     )
# #     data = re.search(pattern,
# #         o, re.DOTALL)
# #     #print(data.groupdict())

