#date: 2025-01-17T16:58:21Z
#url: https://api.github.com/gists/95919dbaed9ea0694972283df71bac92
#owner: https://api.github.com/users/MazenEmadAbdelsattar

# guess a number from 1 to 5
guess = 5
urguess=0
while urguess != guess :
    urguess= int(input("Guess a number from 1 to 5\n"))
    guess -= 1
    if urguess==3:
        print("Thats all right!")
        break
    elif guess== 0 :
        break  
    elif urguess>5:
        print("invalid guess")
        break
    else:
        print(f"Wrong try again, { 'last' if guess == 1 else guess} time is left")