#date: 2026-03-10T17:41:05Z
#url: https://api.github.com/gists/084287c826c33d82631f6e9091c268c5
#owner: https://api.github.com/users/devreyakan

# Kaynak: https://devreyakan.com/roc-ve-auc-egrileri/

from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt


gercek_degerler = [1,1,0,1,0,0,0,1,0,1,1,1]
tahminler = [0.98,0.43,0.3,0.55,0.35,0.2,0.1,0.60,0.03,0.85,0.35,0.58]
tahminler_2 = [0.63,0.54,0.4,0.7,0.4,0.66,0.51,0.4,0.3,0.8,0.71,0.4]

ypo, duyarlilik, esik_degerleri = roc_curve(gercek_degerler,tahminler)
auc_1 = auc(ypo,duyarlilik)

ypo_2, duyarlilik_2, esik_degerleri_2 = roc_curve(gercek_degerler,tahminler_2)
auc_2 = auc(ypo_2,duyarlilik_2)


plt.plot(ypo,duyarlilik,label="Model 1 AUC = %0.3f" %auc_1)
plt.plot(ypo_2,duyarlilik_2,label="Model 2 AUC = %0.3f" %auc_2)

plt.xlabel("Yanlış Pozitif Oranı")
plt.ylabel("Duyarlılık")

plt.legend(loc='lower right')

plt.show()