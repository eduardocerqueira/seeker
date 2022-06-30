#date: 2022-06-30T17:09:29Z
#url: https://api.github.com/gists/50389f82895ed111eef170685f603a39
#owner: https://api.github.com/users/lastshady

from math import *
print("""*********************************************************
DÖNEM SONU NOT HESAPLAYICIYA HOŞ GELDİNİZ!
*********************************************************
""")
print("Bu Uygulamada Bir Dersin Dönem Sonu Sonunda Notunu Hesaplayabilir Ve Harf Notunu Bulabilirsiniz.")

not1 = float(input("Lütfen Birinci Vizeden Aldığınız Notu Giriniz: "))
not2 = float(input("Lütfen İkinci Vizeden Aldığınız Notu Giriniz: "))
not3 = float(input("Lütfen Final Sınavından Aldığınız Notu Giriniz: "))

ort_not = ((not1 * 30) + (not2 * 30) + (not3 * 40)) / 100

liste = {"vize1":not1, "vize2":not2, "final":not3, "ortalama":ort_not}

print("Hesaplanıyor...............")

if (ort_not >= 90):
    print("Not Ortalamanız: {0}, Harf Notunuz: {1}".format(ort_not, "AA"))
elif (ort_not >= 85):
    print("Not Ortalamanız: {0}, Harf Notunuz: {1}".format(ort_not, "BA"))
elif (ort_not >= 80):
    print("Not Ortalamanız: {0}, Harf Notunuz: {1}".format(ort_not, "BB"))
elif (ort_not >= 75):
    print("Not Ortalamanız: {0}, Harf Notunuz: {1}".format(ort_not, "CB"))
elif (ort_not >= 70):
    print("Not Ortalamanız: {0}, Harf Notunuz: {1}".format(ort_not, "CC"))
elif (ort_not >= 65):
    print("Not Ortalamanız: {0}, Harf Notunuz: {1}".format(ort_not, "DC"))
elif (ort_not >= 60):
    print("Not Ortalamanız: {0}, Harf Notunuz: {1}".format(ort_not, "DD"))
elif (ort_not >= 55):
    print("Not Ortalamanız: {0}, Harf Notunuz: {1}".format(ort_not, "FD"))
elif (ort_not < 55):
    print("Not Ortalamanız: {0}, Harf Notunuz: {1}".format(ort_not, "FF"))
else:
    print("HATA!")
