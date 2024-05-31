#date: 2024-05-31T16:42:21Z
#url: https://api.github.com/gists/ed48dff72c74b7b1b445fac1dc0c93d4
#owner: https://api.github.com/users/VedaCPatel

#Make a two-player Rock-Paper-Scissors game.
# (Hint: Ask for player plays (using input), compare them, print out a message of congratulations
# to the winner, and ask if the players want to start a new game)
print("Welcome to the game of Rock,Paper and Scissors!Type 'r','p' and 's' respectively! ")
def multi():
    i = 'start'
    while i != 'end':
        p1 = input("Player 1:")
        p1=p1.lower()
        p2 = input("Player 2:")
        p2=p2.lower()
        if p1 == 'r':
            if p2 == 's':
                print("rock wins!Congrats player 1!")
            elif p2 == 'r':
                print("Draw!")
            else:
                print("Paper wins!Congrats player 2!")
        if p1 == 's':
            if p2 == 'r':
                print("rock wins!Congrats player 2!")
            elif p2 == 's':
                print("Draw!")
            else:
                print("scissor wins!Congrats player 1!")
        if p1 == 'p':
            if p2 == 's':
                print("scissor wins!Congrats player 2!")
            elif p2 == 'p':
                print("Draw!")
            else:
                print("Paper wins!Congrats player 1!")
        i = input("Start game?Type start or end: ")
        i=i.lower()
def comp():
    import random
    i = 'start'
    l=["r","s","p"]
    p2=random.choice(l)
    while i != 'end':
        p1=input("Player: ")
        print("Computer:",p2)
        if p1 == 'r':
            if p2 == 's':
                print("You win!")
            elif p2 == 'r':
                print("You lose!")
            else:
                print("You lose!")
        if p1 == 's':
            if p2 == 'r':
                print("You lose!")
            elif p2 == 's':
                print("Draw!")
            else:
                print("You win!")
        if p1 == 'p':
            if p2 == 's':
                print("You lose!")
            elif p2 == 'p':
                print("Draw!")
            else:
                print("You win!")
        i = input("Start game?Type start or end: ")
q=int(input("For playing in multipayer mode,press 1! and for playing against computer,press 2!:"))
if q==1:
    multi()
else:
    comp()













