#date: 2022-09-30T17:25:53Z
#url: https://api.github.com/gists/f6215bce3aea565c15a613926bde87e0
#owner: https://api.github.com/users/Tharikabalu

weight=int(input('Enter your weight:'))
choose=input('l or k:')
choice=choose.lower()

if choice=='l':
    weight_in_kg=weight*0.454
    print(f'You are {weight_in_kg} kg')
else:
    weight_in_pounds=weight/0.454
    print(f'You are {weight_in_pounds} pounds')


