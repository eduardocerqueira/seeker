#date: 2025-03-11T17:12:26Z
#url: https://api.github.com/gists/d7992f4c28ab4f911913f5d879e19f73
#owner: https://api.github.com/users/koorukuroo

import random

# Define the Monty Hall function
def monty_hall(switch=True):
    """
    Simulates one round of the Monty Hall game.
    
    Parameters:
        switch (bool): If True, the player switches their choice after the host opens a door.
    
    Returns:
        bool: True if the player wins (chooses the car), False otherwise.
    """
    # Step 1: Randomly place the car behind one of the three doors
    prize_door = random.randint(0, 2)

    # Step 2: The player makes an initial random choice
    chosen_door = random.randint(0, 2)

    # Step 3: The host selects a door to open (not the player's choice and not the car)
    doors = [0, 1, 2]
    available_doors = [door for door in doors if door != chosen_door and door != prize_door]
    monty_door = random.choice(available_doors)

    # Step 4: If the player decides to switch, change to the remaining unopened door
    if switch:
        for door in doors:
            if door != chosen_door and door != monty_door:
                chosen_door = door
                break

    # Step 5: Check if the player wins
    return chosen_door == prize_door


# Number of simulations
trials = 10

# List to store results
wins = []

# Run the game multiple times and store results
for _ in range(trials):
    result = monty_hall(switch=True)   # Player switches
    wins.append(result)                          # Append win/loss result (True/False)
# Calculate the probability of winning
win_probability = sum(wins) / len(wins)

# Print results
print(f"Winning probability when switching: {win_probability}")