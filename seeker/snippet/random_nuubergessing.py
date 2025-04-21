#date: 2025-04-21T16:31:30Z
#url: https://api.github.com/gists/ac4837a3b6631f12d48558711d6a51ff
#owner: https://api.github.com/users/bluehatsnak

import random
random_number = random.randint(1, 100)
print("we going to memorize a random number , then you should find this number\n by gessing numbers.\n lets start.")
while True:
    x = int(input("choose a number randomly: "))
    if random_number > x:
        print("More")
    elif random_number < x:
        print("Down")
    elif random_number == x:
        print("you won " + str(random_number) + " is the right number")
        break