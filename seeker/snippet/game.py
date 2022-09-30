#date: 2022-09-30T17:19:32Z
#url: https://api.github.com/gists/0945b86a5d7275772e48f069b2645574
#owner: https://api.github.com/users/Shlok5689

#N=14
v1=int
a=0
print("Please guess a number")
v1=int(input())
while(True):
    if(v1<14):
        print("Oops! Please guess greater number")
        v1 = int(input())
    a=a+1
    if v1>14:
        print("Oops! Please guess small number")
        v1 = int(input())
    a=a+1
    if v1==14:
        print("Wow :) you've guessed correct number.")
    if a>6:
        print("Sorry, you've lost the game")