#date: 2024-02-12T17:06:38Z
#url: https://api.github.com/gists/3a36fbb5ec6683a1db1ac64d15459daf
#owner: https://api.github.com/users/SLB-1986

# Import random pentru generare aleatorie de caractere
import random
# Creez diverse variabile care conțin toate caracterele pe care generatorul le poate utiliza
# Atenție! La litere_mici, litere_mari cât și la simboluri am utilizat caractere din tastatura Germană și Română, nu doar din limba Engleză
litere_mici = "abcdefghijklomnopqrstuvwxyz"+"ßüöä"+"ăîșțâ"
litere_mari = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"+"ÜÖÄ"+"ĂÎȘȚÂ"
numere = "0123456789"
simboluri = "¬`13$%^&*()_+=-[]{};@~#';,./?><|"+'" '+'^`´²³\µ€”¸¨˝´˙`˛°˘^ˇ~`„'
# Parola va fi generată din combinații dintre variabilele care conțin litere mici, mari, numere și simboluri
toate_caracterele = litere_mici + litere_mari + numere +simboluri
# Lungimea parolei selectate de mine este de 15 caractere. Schimbând nr. de mai jos, poți să soliciți să fie din mai multe caractere sau mai puține.
lungime = 15
# Parola generată este simbolizată de ghilimele  simple aici '' care e formată utilizând generatorul random din toate caracterele în lungimea aleasă
parola = ''.join(random.sample(toate_caracterele, lungime))
# Solicit listarea parolei, care e:
print("Parola generată este: ",parola)

# Daca folosești acest generator pentru parole și utilizezi parolele: Nu uita să le salvezi!