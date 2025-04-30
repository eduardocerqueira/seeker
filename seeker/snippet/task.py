#date: 2025-04-30T17:10:20Z
#url: https://api.github.com/gists/cf8091af68b01058d827f56261a37731
#owner: https://api.github.com/users/ininafi

print("Welcome to the treasure island!")
print(r'''
  ____________________________________________________________________
 / \-----     ---------  -----------     -------------- ------    ----\
 \_/__________________________________________________________________/
 |~ ~~ ~~~ ~ ~ ~~~ ~ _____.----------._ ~~~  ~~~~ ~~   ~~  ~~~~~ ~~~~|
 |  _   ~~ ~~ __,---'_       "         `. ~~~ _,--.  ~~~~ __,---.  ~~|
 | | \___ ~~ /      ( )   "          "   `-.,' (') \~~ ~ (  / _\ \~~ |
 |  \    \__/_   __(( _)_      (    "   "     (_\_) \___~ `-.___,'  ~|
 |~~ \     (  )_(__)_|( ))  "   ))          "   |    "  \ ~~ ~~~ _ ~~|
 |  ~ \__ (( _( (  ))  ) _)    ((     \\//    " |   "    \_____,' | ~|
 |~~ ~   \  ( ))(_)(_)_)|  "    ))    //\\ " __,---._  "  "   "  /~~~|
 |    ~~~ |(_ _)| | |   |   "  (   "      ,-'~~~ ~~~ `-.   ___  /~ ~ |
 | ~~     |  |  |   |   _,--- ,--. _  "  (~~  ~~~~  ~~~ ) /___\ \~~ ~|
 |  ~ ~~ /   |      _,----._,'`--'\.`-._  `._~~_~__~_,-'  |H__|  \ ~~|
 |~~    / "     _,-' / `\ ,' / _'  \`.---.._          __        " \~ |
 | ~~~ / /   .-' , / ' _,'_  -  _ '- _`._ `.`-._    _/- `--.   " " \~|
 |  ~ / / _-- `---,~.-' __   --  _,---.  `-._   _,-'- / ` \ \_   " |~|
 | ~ | | -- _    /~/  `-_- _  _,' '  \ \_`-._,-'  / --   \  - \_   / |
 |~~ | \ -      /~~| "     ,-'_ /-  `_ ._`._`-...._____...._,--'  /~~|
 | ~~\  \_ /   /~~/    ___  `---  ---  - - ' ,--.     ___        |~ ~|
 |~   \      ,'~~|  " (o o)   "         " " |~~~ \_,-' ~ `.     ,'~~ |
 | ~~ ~|__,-'~~~~~\    \"/      "  "   "    /~ ~~   O ~ ~~`-.__/~ ~~~|
 |~~~ ~~~  ~~~~~~~~`.______________________/ ~~~    |   ~~~ ~~ ~ ~~~~|
 |____~jrei~__~_______~~_~____~~_____~~___~_~~___~\_|_/ ~_____~___~__|
 / \----- ----- ------------  ------- ----- -------  --------  -------\
 \_/__________________________________________________________________/
''')
print("There is a hidden treasure worth of 1000kg gold in this island."
      "However, it is located on the top of mountain and there is a dragon keeping the area")
consent = input("Do you want to join to hunt this treasure and risk your life? Answer with Y or N\n")
if consent == "Y":
    print("Welcome warrior! Your adventure is start here, ")
else:
    print("Good bye troops! Go home and get ready for another adventure ahead!")

print("In this journey full of adventure, you will find unexpected things you never expected"
      "To avoid bad things happen, you need to inform us your background")
name = input("What is your name?\n")
print(f"Welcome " + name +"!")
age = int(input("How old are you?\n"))
if age >= 18:
    print("Great! You are eligible to join this hunting")
else:
    print("You can come back again when you are 18! Focus on your study kid!")

swim = input("Are you able to swim? Answer with Y or N\n")
if swim == "Y":
    print("There will be a nice lake near the mountain, you can enjoy the crystal clear water and nice view")
else:
    print("Be careful around the lake!")
climb = input("Are you able to climb? Answer with Y or N\n")
if climb == "Y":
    print("There might be a beast out there, climb the trees to escape!")
else:
    print("Watch out! Find the hiding from the beast")

print("Stage 1")
ready = input("Are you ready?")
if ready == "Y":
    print(r'''
  *******************************************************************************
          |                   |                  |                     |
 _________|________________.=""_;=.______________|_____________________|_______
|                   |  ,-"_,=""     `"=.|                  |
|___________________|__"=._o`"-._        `"=.______________|___________________
          |                `"=._o`"=._      _`"=._                     |
 _________|_____________________:=._o "=._."_.-="'"=.__________________|_______
|                   |    __.--" , ; `"=._o." ,-"""-._ ".   |
|___________________|_._"  ,. .` ` `` ,  `"-._"-._   ". '__|___________________
          |           |o`"=._` , "` `; .". ,  "-._"-._; ;              |
 _________|___________| ;`-.o`"=._; ." ` '`."\ ` . "-._ /_______________|_______
|                   | |o ;    `"-.o`"=._``  '` " ,__.--o;   |
|___________________|_| ;     (#) `-.o `"=.`_.--"_o.-; ;___|___________________
____/______/______/___|o;._    "      `".o|o_.--"    ;o;____/______/______/____
/______/______/______/_"=._o--._        ; | ;        ; ;/______/______/______/_
____/______/______/______/__"=._o--._   ;o|o;     _._;o;____/______/______/____
/______/______/______/______/____"=._o._; | ;_.--"o.--"_/______/______/______/_
____/______/______/______/______/_____"=.o|o_.--""___/______/______/______/____
/______/______/______/______/______/______/______/______/______/______/_____ /
*******************************************************************************
''')
else:
    print("you typed the wrong answer")

print("Stage 1")
print("Your mission is to find the treasure.")
print("Look at the map! Now you are on the gate.")
path1 = input("Choose the path you want to go. left or right\n")
if path1 == "left":
    print("Go ahead!" + name + "One step closer to the gold!")
else:
    print(name + "You chose the wrong path. You are eliminated!")

print("Your water bottle is almost empty! You need to go to the lake"
      "However, you need to calculate how long it takes to be there."
      "Follow the red mark on the trees and see the instructions")

number = int(input("What are the number that you see on the mark?\n"))
distance = 30 - number
print(f"Your distance to the lake now is " + str(distance) + "km")