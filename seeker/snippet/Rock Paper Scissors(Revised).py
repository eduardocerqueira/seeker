#date: 2021-08-31T13:06:35Z
#url: https://api.github.com/gists/c79c38e318037f9adba22d226f1e8436
#owner: https://api.github.com/users/TruthyCode

from random import randint


options = ["r", "s", "p"]
abbr_option = {
    "r": "Rock",
    "s": "Scissors",
    "p": "Paper"
}
player_win = 0
computer_win = 0
looped = False

while True:
    computer_choice = options[randint(0, 2)]
    player_choice = input("""\
Type:
r - Rock
p - paper
s - Scissors
q - to Quit

>>> """) if not looped else input(">>>")
    if player_choice == "q":
        print(f"\nThank You for playing.\n")
        if looped:
            print(f"""\
Wins : {player_win}
Losses: {computer_win}
Score: {(player_win - computer_win) if player_win > computer_win else 0}""")
        exit()
    if player_choice in options:
        looped = True
        if (player_choice == options[0] and computer_choice == options[1]) or \
                (player_choice == options[1] and computer_choice == options[2]) or \
                (player_choice == options[2] and computer_choice == options[0]):
            print(f"Yey! You Won!\nThe Computer Chose {abbr_option[computer_choice]}")
            player_win += 1
            continue

        if player_choice == computer_choice:
            print(f"Both chose {abbr_option[computer_choice]}.\nIt was a tie!")
            continue

        else:
            print(f"Oops! The Computer Chose {abbr_option[computer_choice]}.\nThe Computer Won :(")
            computer_win += 1
            continue

