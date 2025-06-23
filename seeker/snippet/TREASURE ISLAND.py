#date: 2025-06-23T17:04:50Z
#url: https://api.github.com/gists/4bfb7e1f7633149b8c4e69b024a39aa0
#owner: https://api.github.com/users/072zahra

print("WELCOME TO TREASURE ISLAND")
print("YOUR MISSION TO FIND THE TREASURE")
choice1 = input('you\ are at crossroad, where u want to go?type"left"or "right"')

if choice1 =="left":
    choice2 =input('you\ have come to lake.type"wait"to wait for boat.type"swim"to swim across').lower()
    if choice2=="wait":
        choice3=input("you arrive at island unharmed. there is house with three doors.yellow,red and blue.which colour u choose?").lower()
        if choice3=="red":
            print("room full of fire.GAME OVER")
        elif choice3=="yellow":
            print("you found treasure")
        elif choice3=="blue":
            print("you enter in room of hell.game over")
        else:
            print("you enter room where nothing exists .GO TO HELL")
    else:
        print("you attacked by trout. have a good ending")
else:
    print("YOU FELL IN HOLE. GAME OVER")