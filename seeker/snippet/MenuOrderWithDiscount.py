#date: 2022-10-07T17:14:26Z
#url: https://api.github.com/gists/4356e6b65ecd2ae1c33c37f9d96c1c39
#owner: https://api.github.com/users/clarissecasicarino

discount_child = 0.50
discount_senior = 0.25
penalty = -0.25
coke = 1.00
dosa = 2.50
pizza = 2.25
taco = 1.50
tea = 1.00

print("---- Task 1: Simple Order ----")
print("**Select menu item**")
print("(1) \t Coke \t [$1.00]")
print("(2) \t Dosa \t [$2.50]")
print("(3) \t Pizza \t [$2.25]")
print("(4) \t Taco \t [$1.50]")
print("(5) \t Tea \t [$1.00]")

selectOrder = int(input("Selection:"))
print("**Discount**")
print("(C) Child [under 18] (50% discount)")
print("(A) Adult [18-64]")
print("(S) Senior [65+] (25% discount)")

selectAge = input("Selection Age:")

if selectOrder == 1:
    print("Amount" + "\t$" + "\t{:.2f}".format(coke))
    if selectAge.upper() == "C" or selectAge.lower() == "c":
        getCokeChildDiscount = coke * discount_child
        print("Disc" + "\t$" + "\t{:.2f}".format(getCokeChildDiscount))
        print("-----------------")
        CokeChildTotal = coke - getCokeChildDiscount
        print("Total" + "\t$" + "\t{:.2f}".format(CokeChildTotal))

    elif selectAge.upper() == "S" or selectAge.lower() == "s":
        getCokeSeniorDiscount = coke * discount_senior
        print("Disc" + "\t$" + "\t{:.2f}".format(getCokeSeniorDiscount))
        print("-----------------")
        CokeSeniorTotal = coke - getCokeSeniorDiscount
        print("Total" + "\t$" + "\t{:.2f}".format(CokeSeniorTotal))

    elif selectAge.upper() == "A" or selectAge.lower() == "a":
        getCokeAdultPrice = coke * 0
        print("Disc" + "\t$" + "\t{:.2f}".format(getCokeAdultPrice))
        print("-----------------")
        CokeAdultTotal = coke - getCokeAdultPrice
        print("Total" + "\t$" + "\t{:.2f}".format(CokeAdultTotal))

    elif selectAge != "A" or selectAge != "a" or selectAge != "C" or selectAge != "c" or selectAge != "S" or selectAge != "s":
        getCokePenalty = coke * penalty
        print("Disc" + "\t$" + "\t{:.2f}".format(getCokePenalty))
        print("-----------------")
        CokeAdultTotal = coke - getCokePenalty
        print("Total" + "\t$" + "\t{:.2f}".format(CokeAdultTotal))

elif selectOrder == 2:
    print("Amount" + "\t$" + "\t{:.2f}".format(dosa))
    if selectAge.upper() == "C" or selectAge.lower() == "c":
        getDosaChildDiscount = dosa * discount_child
        print("Disc" + "\t$" + "\t{:.2f}".format(getDosaChildDiscount))
        print("-----------------")
        DosaChildTotal = dosa - getDosaChildDiscount
        print("Total" + "\t$" + "\t{:.2f}".format(DosaChildTotal))

    elif selectAge.upper() == "S" or selectAge.lower() == "s":
        getDosaSeniorDiscount = dosa * discount_senior
        print("Disc" + "\t$" + "\t{:.2f}".format(getDosaSeniorDiscount))
        print("-----------------")
        DosaSeniorTotal = dosa - getDosaSeniorDiscount
        print("Total" + "\t$" + "\t{:.2f}".format(DosaSeniorTotal))

    elif selectAge.upper() == "A" or selectAge.lower() == "a":
        getDosaAdultPrice = dosa * 0
        print("Disc" + "\t$" + "\t{:.2f}".format(getDosaAdultPrice))
        print("-----------------")
        DosaAdultTotal = dosa - getDosaAdultPrice
        print("Total" + "\t$" + "\t{:.2f}".format(DosaAdultTotal))

    elif selectAge != "A" or selectAge != "a" or selectAge != "C" or selectAge != "c" or selectAge != "S" or selectAge != "s":
        getDosaPenalty = dosa * penalty
        print("Disc" + "\t$" + "\t{:.2f}".format(getDosaPenalty))
        print("-----------------")
        DosaAdultTotal = dosa - getDosaPenalty
        print("Total" + "\t$" + "\t{:.2f}".format(DosaAdultTotal))

elif selectOrder == 3:
    print("Amount" + "\t$" + "\t{:.2f}".format(pizza))
    if selectAge.upper() == "C" or selectAge.lower() == "c":
        getPizzaChildDiscount = pizza * discount_child
        print("Disc" + "\t$" + "\t{:.2f}".format(getPizzaChildDiscount))
        print("-----------------")
        PizzaChildTotal = pizza - getPizzaChildDiscount
        print("Total" + "\t$" + "\t{:.2f}".format(PizzaChildTotal))

    elif selectAge.upper() == "S" or selectAge.lower() == "s":
        getPizzaSeniorDiscount = pizza * discount_senior
        print("Disc" + "\t$" + "\t{:.2f}".format(getPizzaSeniorDiscount))
        print("-----------------")
        PizzaSeniorTotal = pizza - getPizzaSeniorDiscount
        print("Total" + "\t$" + "\t{:.2f}".format(PizzaSeniorTotal))

    elif selectAge.upper() == "A" or selectAge.lower() == "a":
        getPizzaAdultPrice = pizza * 0
        print("Disc" + "\t$" + "\t{:.2f}".format(getPizzaAdultPrice))
        print("-----------------")
        PizzaAdultTotal = pizza - getPizzaAdultPrice
        print("Total" + "\t$" + "\t{:.2f}".format(PizzaAdultTotal))

    elif selectAge != "A" or selectAge != "a" or selectAge != "C" or selectAge != "c" or selectAge != "S" or selectAge != "s":
        getPizzaPenalty = pizza * penalty
        print("Disc" + "\t$" + "\t{:.2f}".format(getPizzaPenalty))
        print("-----------------")
        PizzaAdultTotal = pizza - getPizzaPenalty
        print("Total" + "\t$" + "\t{:.2f}".format(PizzaAdultTotal))

elif selectOrder == 4:
    print("Amount" + "\t$" + "\t{:.2f}".format(taco))
    if selectAge.upper() == "C" or selectAge.lower() == "c":
        getTacoChildDiscount = taco * discount_child
        print("Disc" + "\t$" + "\t{:.2f}".format(getTacoChildDiscount))
        print("-----------------")
        TacoChildTotal = taco - getTacoChildDiscount
        print("Total" + "\t$" + "\t{:.2f}".format(TacoChildTotal))

    elif selectAge.upper() == "S" or selectAge.lower() == "s":
        getTacoSeniorDiscount = taco * discount_senior
        print("Disc" + "\t$" + "\t{:.2f}".format(getTacoSeniorDiscount))
        print("-----------------")
        TacoSeniorTotal = taco - getTacoSeniorDiscount
        print("Total" + "\t$" + "\t{:.2f}".format(TacoSeniorTotal))

    elif selectAge.upper() == "A" or selectAge.lower() == "a":
        getTacoAdultPrice = taco * 0
        print("Disc" + "\t$" + "\t{:.2f}".format(getTacoAdultPrice))
        print("-----------------")
        TacoAdultTotal = taco - getTacoAdultPrice
        print("Total" + "\t$" + "\t{:.2f}".format(TacoAdultTotal))

    elif selectAge != "A" or selectAge != "a" or selectAge != "C" or selectAge != "c" or selectAge != "S" or selectAge != "s":
        getTacoPenalty = taco * penalty
        print("Disc" + "\t$" + "\t{:.2f}".format(getTacoPenalty))
        print("-----------------")
        TacoAdultTotal = taco - getTacoPenalty
        print("Total" + "\t$" + "\t{:.2f}".format(TacoAdultTotal))

elif selectOrder == 5:
    print("Amount" + "\t$" + "\t{:.2f}".format(tea))
    if selectAge.upper() == "C" or selectAge.lower() == "c":
        getTeaChildDiscount = tea * discount_child
        print("Disc" + "\t$" + "\t{:.2f}".format(getTeaChildDiscount))
        print("-----------------")
        TeaChildTotal = tea - getTeaChildDiscount
        print("Total" + "\t$" + "\t{:.2f}".format(TeaChildTotal))

    elif selectAge.upper() == "S" or selectAge.lower() == "s":
        getTeaSeniorDiscount = tea * discount_senior
        print("Disc" + "\t$" + "\t{:.2f}".format(getTeaSeniorDiscount))
        print("-----------------")
        TeaSeniorTotal = tea - getTeaSeniorDiscount
        print("Total" + "\t$" + "\t{:.2f}".format(TeaSeniorTotal))

    elif selectAge.upper() == "A" or selectAge.lower() == "a":
        getTeaAdultPrice = tea * 0
        print("Disc" + "\t$" + "\t{:.2f}".format(getTeaAdultPrice))
        print("-----------------")
        TeaAdultTotal = tea - getTeaAdultPrice
        print("Total" + "\t$" + "\t{:.2f}".format(TeaAdultTotal))

    elif selectAge != "A" or selectAge != "a" or selectAge != "C" or selectAge != "c" or selectAge != "S" or selectAge != "s":
        getTeaPenalty = tea * penalty
        print("Disc" + "\t$" + "\t{:.2f}".format(getTeaPenalty))
        print("-----------------")
        TeaAdultTotal = tea - getTeaPenalty
        print("Total" + "\t$" + "\t{:.2f}".format(TeaAdultTotal))

else:
    print("Invalid Selection - setting amount to $0")
    setAmount = 0.00
    setDisc = 0.00
    setTotal = 0.00
    print("Amount" + "\t$" + "\t{:.2f}".format(setDisc))
    print("Disc" + "\t$" + "\t{:.2f}".format(setDisc))
    print("-----------------")
    print("Total" + "\t$" + "\t{:.2f}".format(setTotal))