#date: 2022-03-14T17:04:47Z
#url: https://api.github.com/gists/fe4ea790e976e45b1479195b0cce3e0d
#owner: https://api.github.com/users/ShubhamKalsekar

import random

num = random.randint(1,100)
guess = None
count=1
while guess != num:
    # count = + 1
    guess = input("Guess a Number between 1 to 100: \n")
    guess = int(guess)
    if num == guess:

        print("Congratulations ,You did it in ", count, " try")

        # Once guessed, loop will break

        break

    elif num > guess:

        print("You guessed too small!")

    elif num < guess:

        print("You Guessed too high!")

    count = count + 1