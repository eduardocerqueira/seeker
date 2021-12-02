#date: 2021-12-02T16:50:17Z
#url: https://api.github.com/gists/a089c92d43e2bdb122f75a92b8d6de7a
#owner: https://api.github.com/users/AnabelSalomone

# Ecrire un programme qui demande l'age et si on veut du popcorn
# et afficher le prix.
# if > 18 === 10.5
# if 18 < === 7.5
# popcorn 4.5

popcorn_price = 4.5
adult_price = 10.5
young_price = 7.5
valid_age = False
valid_input = False

total = adult_price  

while valid_age == False: 
    try:
        age = int(input("Quel est votre age?"))
        if age < 18 : total = young_price
        valid_age = True
    except Exception:
        print("Invalid input")

while valid_input==False:
    popcorn=input("Voulez-vous du popcorn?")
    if popcorn == "Oui":
        total += popcorn_price
        valid_input=True
    elif popcorn == "Non":
        valid_input = True
    else:
        print("Invalid input")

print("Total: ", total)