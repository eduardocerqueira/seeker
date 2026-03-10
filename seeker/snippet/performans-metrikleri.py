#date: 2026-03-10T17:35:06Z
#url: https://api.github.com/gists/370739c899d1d2d45ce7efc90e439f47
#owner: https://api.github.com/users/devreyakan

from sklearn.metrics import r2_score

gercek_degerler = [10,11.2,13,20,9,8.5,7.3]

iyi_tahminler = [9.9,11.2,12.5,19.8,9.1,8.23,7.2]
kotu_tahminler = [15,9,17,15.3,5.5,6.3,11.5]

print("İyi yapılmış tahminlerin R^2 skoru : ", r2_score(gercek_degerler,iyi_tahminler))
print("Kötü yapılmış tahminlerin R^2 skoru : ", r2_score(gercek_degerler,kotu_tahminler))