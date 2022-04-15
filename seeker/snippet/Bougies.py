#date: 2022-04-15T16:50:25Z
#url: https://api.github.com/gists/cd0f916083dd2c5760b0b1391d628449
#owner: https://api.github.com/users/Dan-Lysight

# Bougies.py
# Calcul le nombre de bougies d'anniversaire soufflées?
# Dan et Alain, le 18 juin 2021

age=int(input("Quel âge avez-vous: "))
bougiestot=0
for bougie in range (1, age+1):
    bougiestot = bougiestot+bougie

if (bougiestot==1):
    print (str(age)+" ans : vous avez soufflé "+str(bougiestot)+" bougie.")
else:
    print (str(age)+" ans : vous avez soufflé "+str(bougiestot)+" bougies.")
