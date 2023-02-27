#date: 2023-02-27T17:02:11Z
#url: https://api.github.com/gists/7f3c47460ee2df678a59c81a65d11951
#owner: https://api.github.com/users/Aamelio22

# Ascii art for rock, paper, and scissors.

rock = '''
    _______
---'   ____)
      (_____)
      (_____)
      (____)
---.__(___)
'''

paper = '''
    _______
---'   ____)____
          ______)
          _______)
         _______)
---.__________)
'''

scissors = '''
    _______
---'   ____)____
          ______)
       __________)
      (____)
---.__(___)
'''

# Importing random  library

import random

# Setting a random number from 0 to 2 to be put a variable.

comp_choice = random.randint(0, 2)

# Creating counter for results to be stored in variables.

w = 0
l = 0
d = 0

# Setting game status to run

game_status = "run"

# Establishing one while loop to replay if user wishes and one to negate user inout error.
while game_status == "run":
    while True:
        player_choice = int(input("\nWhat do you choose? Type 0 for Rock, 1 for Paper, or 2 for Scissors.\n"))
        if (player_choice) not in range(0, 3):
            print("\nError, please type a valid number. (0, 1, 2)")
            continue
        else:
            break
            # Print Ascii art based on user input and setting game outcome status to a variable.

    if player_choice == 0:
        print(f"{rock}")
        if comp_choice == 0:
            outcome = "d"
        elif comp_choice == 1:
            outcome = "l"
        else:
            outcome = "w"
    elif player_choice == 1:
        print(f"{paper}")
        if comp_choice == 0:
            outcome = "w"
        elif comp_choice == 1:
            outcome = "d"
        else:
            outcome = "l"
    else:
        print(f"{scissors}")
        if comp_choice == 0:
            outcome = "l"
        elif comp_choice == 1:
            outcome = "w"
        else:
            outcome = "d"

            # Print computer choice and then print game outcome.

    if comp_choice == 0:
        print(f"\nComputer  chose:\n{rock}\n")
    elif comp_choice == 1:
        print(f"\nComputer chose:\n{paper}\n")
    else:
        print(f"\nComputer chose:\n{scissors}\n")

    if outcome == "w":
        print("You win!\n")
        w += 1
    elif outcome == "l":
        print("You lose!\n")
        l += 1
    else:
        print("It's a draw!\n")
        d += 1
    while True:
        choice = input("Whould you like to try again? (y/n)\n\n").lower()
        sep = [let for let in choice]
        if "y" in sep:
            break
        elif "n" in sep:
            print(f"\nGoodbye! Thanks for playing!\n\nYour score tally is:\nWins|{w}\nLosses|{l}\nDraws|{d}")
            game_status = "end"
            break
        else:
            print("Error, please type y or n")