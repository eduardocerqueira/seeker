#date: 2022-12-22T17:01:16Z
#url: https://api.github.com/gists/f289e555815859cd544b25fbc0fb4dd0
#owner: https://api.github.com/users/mathiashoeld

import random
import math
import time

def distance_to_mean(mean, actual):
    return math.sqrt((mean-actual)*(mean-actual))*100/mean

def choose_winner_bingo(names):
    print("And the winner is...")
    print("... Drumroll ...")
    time.sleep(10)
    print("")
    
    return names[random.randint(0,len(names)-1)]


def test_random_number_generator(names, iterations_count):
    print("Um den Random Number Generator zu testen,")
    print(f"lassen wir ihn erstmal {iterations_count} mal laufen")
    print("und checken, ob auch jede:r eine faire Chance hat!")
    print("")

    names_count = {}
    for name in names:
        names_count[name] = 0

    for i in range(0,iterations_count):
        random_name = names[random.randint(0,len(names)-1)]
        names_count[random_name] += 1

    mean = iterations_count/len(names)

    print(f"Durchschnittswert: {mean}")
    print("")

    for name in names_count.keys():
        print(f"Name: {name}"), 
        print(f"\tAnzahl: {names_count[name]}")
        print(f"\tAbweichung vom Mittelwert: {distance_to_mean(mean, names_count[name])}%")


def print_welcome_message(names):
    print("---------------------")
    print(" Willkommen bei der Weihnachtsbingo Sieger:innen Ehrung")
    print("---------------------")
    print("Folgende Leute haben erfolgreich teilgenommen: ")
    for name in names:
        print(name)
    print("---------------------")



if __name__ == "__main__":
    names = [
        "Moda Gal",
        "Kasche Rar",
        "Vaj Erinnen",
        "Liam Hunt",
        "Dorn Dahn",
        "Bic Keithel",
        "Rhynna Horle",
        "Gaeriel Staven",
        "Mala Deccol",
        "Tifa Dreytila",
        "Kana Daagh",
        "Eriobea Eronoss"
    ]


    iterations_count = 100000000

    print_welcome_message(names)
    test_random_number_generator(names, iterations_count)

    print(f"{choose_winner_bingo(names).upper()}!") 
    
