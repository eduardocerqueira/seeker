#date: 2022-03-04T16:56:16Z
#url: https://api.github.com/gists/f360901c0a052be85abe07fc27d3eece
#owner: https://api.github.com/users/kashishAhuja

a=int(input("enter the minimum range"))
b=int(input("enter the maximum range"))
attempts=0
attempt=0
import random
while attempts<10:
  c=(random.randint(a,b))
  attempts=attempts+1
  player1=int(input("guess a number"))
  if player1==c:
    print("wow,your guess is correct")
    break
  elif player1>c:
    print("opps,you guessed a higher number")
  else:
    print("opps,you guessed a lower number")
  print(f"{10 - attempts} attempts left!!!")
s=int(input("enter the minimum range"))
t=int(input("enter the maximum range"))
while attempt<10:
  d=(random.randint(s,t))
  attempt=attempt+1
  player2=int(input("guess a number"))
  if player2==d:
    print("wow,your guess is correct")
    break
  elif player2>d:
    print("opps,you guessed a higher number")
  else:
    print("opps,you guessed a lower number")
  print(f"{10 - attempt} attempts left!!!")
