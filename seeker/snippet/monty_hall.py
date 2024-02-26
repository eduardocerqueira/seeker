#date: 2024-02-26T17:05:11Z
#url: https://api.github.com/gists/b1fe4aca166e26efec9e25acd0f12a85
#owner: https://api.github.com/users/sajad-sadra

#!/usr/bin/python3
from random import randint

# just change these 3 parameter
number_of_doors = 100
number_of_experiment = 999999
change_door_flag = True


seperator = "==================================================================="
print("Staring", number_of_experiment, " MontyHall paradox experiment.")
print("Each of them has", number_of_doors, " doors.")
verb = "will not"
if (change_door_flag):
    verb = "will"
print("And the player", verb, "change their choice.")
print(seperator)

win_counter = 0
for experiment in range(number_of_experiment):
    doors = []
    for i in range(number_of_doors):
        doors.append(0)
    gift_door = randint(0, number_of_doors-1)
    doors[gift_door] = 1

    player_choice = randint(0, number_of_doors-1)

    monty_holl_choice = randint(0, number_of_doors-1)
    while((monty_holl_choice == player_choice)):
        monty_holl_choice = randint(0, number_of_doors-1)
    
    if (change_door_flag):
        player_choice = monty_holl_choice

    game_reult = "Lost"
    if (player_choice == gift_door):
        game_reult = "Win"
        win_counter += 1
    
    print("Game gift_door was", gift_door, "player choice was", player_choice, "montyHall choice was", monty_holl_choice, "and player", game_reult)

print(seperator)
success_rate = (win_counter/number_of_experiment) * 100
print("Player won", win_counter, "times.")
print("Success Rate:", success_rate, "%")