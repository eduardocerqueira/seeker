#date: 2022-02-24T16:53:20Z
#url: https://api.github.com/gists/7843e2c6e0b1aec123eb1fb2e0a88ec7
#owner: https://api.github.com/users/Pritchard26

import random
import os
from traceback import print_tb
def clear():
	# İşletim Sistemi Windows ise
	if os.name == 'nt':
	    _ = os.system('cls')
	# İşletim sistemi Linux veya Mac ise

cevap="e"
while cevap=="e":
	can=4
	kazanma=0
	cevap="h"
	liste=["araba","elma"]#kelimelerin olduğu liste ekleyebilirsiniz elbette 
	rastgele_kelime=random.choice(liste)#listeden rastgele bir kelime seçer
	harf_sayısı = len(rastgele_kelime)#kelimenin harf sayısı
	liste = []#tireler için liste
	kelimelistesi=[]#bu listeyi kelimenin harflerini tek tek listeye ekleyip oradan aldığımız harf ile karşılaştırılmasını sağlamak amacıyla açtım
	for i in range(harf_sayısı):#mantığı harf sayısı kadar listeye tire eklemesi
		liste=liste+ ["_ "]
	#print("*"*5, " Adam Asmaca ", "*"*5)	
	#print(liste , can ,"Canınız kaldı.")	
	for i in rastgele_kelime:# i kelimedeki 1 tane harfi alıp onu kelime listesine ekliyor oradan karşılaştırmak adına
		kelimelistesi=kelimelistesi+[i]
	while can>=1:
		print("*"*5, " Adam Asmaca ", "*"*5)
		print(liste , can ,"Canınız kaldı.")
		if can==3:
			print(" "*26," ___\n                              |\n                              O ")
		elif can==2:
			print(" "*26," ___\n                              |\n                              O\n                              | ")				
		elif can==1:
			print(" "*26," ___\n                              |\n                              O\n                             /|\ ")	
		alınan_harf = input("Harf giriniz:.. ")
		if alınan_harf not in kelimelistesi:#eğer aldığımız harf yoksa buraya gelip canı azaltıyor ve adamı 1 kademe asıyor
			can-=1
			print("Bilemediniz", can, "Canınız kaldı..")
			input("")
			if can==3:
				print(" "*26," ___\n                              |\n                              O ")
			elif can==2:       
				print(" "*26," ___\n                              |\n                              O\n                              | ")					
			elif can==1:
				print(" "*26," ___\n                              |\n                              O\n                             /|\ ")	
			clear()
			continue			
	#	print(liste)
	#	print(kelimelistesi)
		for i in range(harf_sayısı):# kelime listesindeki elemanları tek tek i sayesinde harf e ekliyoruz 
			harf = kelimelistesi[i]
			if alınan_harf == harf: # eğer kelimelistesindeki eleman ile aldığımız harf eşit ise o harfin yerine yıldız koyuyor 
				kelimelistesi[i] = "*"
				liste[i]= alınan_harf#listede o sıradaki tirenin yerine alınan harfi ekliyor 
				kazanma+=1
			continue		
		input("Bildiniz")
		print(liste)
		if kazanma == harf_sayısı:
			cevap=input("Kazandınızzz ( Devam etmek için e/h )")
			if cevap== "e":
				clear()
			else:
				break
		clear()