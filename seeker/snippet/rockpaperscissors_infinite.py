#date: 2024-08-20T17:06:04Z
#url: https://api.github.com/gists/a85000e3c4e4d20d491aa5baf55522bb
#owner: https://api.github.com/users/blockchain200

import random

def invalid_item():
    print("Invalid item. Terminating...")

items = {
    'A': "asteroid",
    'B': "ball",
    'C': "cake",
    'D': "door",
    'E': "elephant",
    'F': "finger",
    'G': "gun",
    'H': "hacker",
    'I': "igloo",
    'J': "janitor",
    'K': "knowledge",
    'L': "lol",
    'M': "mettaton",
    'N': "null",
    'O': "opp",
    'P': "paper",
    'Q': "croissant",
    'R': "rock",
    'S': "scissors",
    'T': "taxonomy",
    'U': "ukelele",
    'V': "vigilant",
    'W': "wash-your-hands",
    'X': "x? what kinda word starts with the letter x?",
    'Y': "yes",
    'Z': "zebra"
}

print("\n".join(f"{k} - {v}" for k, v in items.items()))

current_item_rand = random.choice(["ROCK", "PAPER", "SCISSORS"])

print(f"{current_item_rand} vs: ", end="")

input_char = input().strip().upper()

match input_char:
    case 'R':
        match current_item_rand:
            case "ROCK":
                print("ROCK and ROCK... They both got a rock-er.")
            case "PAPER":
                print("PAPER suffocates the soul out of ROCK!")
            case "SCISSORS":
                print("ROCK smashes the shit out of SCISSORS!")
    
    case 'P':
        match current_item_rand:
            case "ROCK":
                print("PAPER suffocates the soul out of ROCK!")
            case "PAPER":
                print("PAPER and PAPER... They kiss.")
            case "SCISSORS":
                print("SCISSORS slice up PAPER into bits.")
    
    case 'S':
        match current_item_rand:
            case "ROCK":
                print("ROCK smashes the shit out of SCISSORS!")
            case "PAPER":
                print("SCISSORS slice up PAPER into bits.")
            case "SCISSORS":
                print("SCISSORS and SCISSORS... They made out.")
    
    case 'A':
        match current_item_rand:
            case "ROCK":
                print("ASTEROID crushes ROCK with its mighty force!")
            case "PAPER":
                print("PAPER summons Death By A Thousand Cuts and cuts ASTEROID to death!")
            case "SCISSORS":
                print("SCISSORS cut ASTEROID in half, but got shattered in the process. It's a tie.")
    
    case 'B':
        match current_item_rand:
            case "ROCK":
                print("BALL bounced up and down, making ROCK vomit from the nausea.")
            case "PAPER":
                print("PAPER was able to catch BALL while it was bouncing!")
            case "SCISSORS":
                print("SCISSORS popped BALL like a balloon!")
    
    case 'C':
        match current_item_rand:
            case "ROCK":
                print("ROCK destroys the CAKE!")
            case "PAPER":
                print("PAPER covers the CAKE, but it's too delicious to resist!")
            case "SCISSORS":
                print("SCISSORS slice the CAKE into perfect pieces!")
    
    case 'D':
        match current_item_rand:
            case "ROCK":
                print("DOOR made a squeaky sound when it opened and ruptured ROCK's ears!")
            case "PAPER":
                print("PAPER slid under the DOOR and escaped!")
            case "SCISSORS":
                print("SCISSORS drew (scratched) a smiley face on the DOOR, and DOOR got embarrassed!")
    
    case 'E':
        match current_item_rand:
            case "ROCK":
                print("ELEPHANT turned ROCK into dust by stepping on it!")
            case "PAPER":
                print("PAPER got lost in the ELEPHANT's hair molecules!")
            case "SCISSORS":
                print("ELEPHANT's nails were groomed by SCISSORS!")
    
    case 'F':
        match current_item_rand:
            case "ROCK":
                print("FINGER went into ROCK's ass!")
            case "PAPER":
                print("FINGER stabbed PAPER and left a big hole!")
            case "SCISSORS":
                print("SCISSORS cut off FINGER's head like a carrot!")
    
    case 'G':
        match current_item_rand:
            case "ROCK":
                print("ROCK destroys the gun!")
            case "PAPER":
                print("GUN shoots through the PAPER!")
            case "SCISSORS":
                print("SCISSORS deflected the GUN's bullet!")
    
    case 'H':
        match current_item_rand:
            case "ROCK":
                print("ROCK destroyed HACKER's computer!")
            case "PAPER":
                print("PAPER climbed into HACKER's USB drive!")
            case "SCISSORS":
                print("HACKER hacked SCISSORS and made it go ballistic!")
    
    case 'I':
        match current_item_rand:
            case "ROCK":
                print("ROCK isn't used to cold weather, so it got frozen by IGLOO!")
            case "PAPER":
                print("PAPER rode the icy cold winds, escaping from IGLOO!")
            case "SCISSORS":
                print("IGLOO trapped SCISSORS inside his territory!")
    
    case 'J':
        match current_item_rand:
            case "ROCK":
                print("JANITOR cleaned the dust off of ROCK, leaving ROCK embarrassed.")
            case "PAPER":
                print("PAPER cleaned JANITOR, leaving his self-esteem at an all-time low.")
            case "SCISSORS":
                print("SCISSORS sliced JANITOR's mop, making him go cry in the corner.")
    
    case 'K':
        match current_item_rand:
            case "ROCK":
                print("KNOWLEDGE outsmarted ROCK's dumb brain!")
            case "PAPER":
                print("PAPER modeled a 3D diagram of the 4th dimension and outsmarted KNOWLEDGE!")
            case "SCISSORS":
                print("SCISSORS cut through KNOWLEDGE's brain cells!")
    
    case 'M':
        match current_item_rand:
            case _:
                print("Mettaton.")
    
    case _:
        invalid_item()
