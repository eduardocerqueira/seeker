#date: 2022-12-14T16:51:06Z
#url: https://api.github.com/gists/6bf117e28af23691050985bcc9dc09d6
#owner: https://api.github.com/users/ARAGORN1453

dizi=input("bir dizi girin:")
liste=[]
for i in dizi:
    i=int(i)
    if i%2==0:
        liste.append(i)
print(liste)

eb=max(liste)
ek=min(liste)
a=eb+ek
ortalama=a/2
print(ortalama)


