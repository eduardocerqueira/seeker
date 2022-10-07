#date: 2022-10-07T17:19:53Z
#url: https://api.github.com/gists/dd68fb3df8ef598539b9d8dc97027c45
#owner: https://api.github.com/users/edanursunay

def add(*numbers):     #birden fazla değer alabilmek için * kullandık
    total = 0
    for i in numbers:
        total += i
    print("Toplam: ", total)

add() # 0 argument
add(29, 36, 7, 15) # 4 arguments
add(98.3, 2, 27) # 3 arguments