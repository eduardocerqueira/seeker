#date: 2022-01-26T17:11:36Z
#url: https://api.github.com/gists/60317d153d7e17d25670168aa26749e2
#owner: https://api.github.com/users/overas

# Enkel kode til skritteller

from microbit import *

steps = 0  # Antall steg nullstilles

while True:  # En løkke som kjører til den blir avbrutt

    if (
        accelerometer.current_gesture() == "shake"
    ):  # Dersom microbiten ristes, skjer følgende:

        steps += 1  # Antall steg øker med 1
    if button_a.is_pressed():  # Dersom knapp a er blitt trykket på, skjer følgende:

        display.show(steps)  # Antall steg vises på displayet

        sleep(500)  # Løkka settes på pause i 500 millisekunder
    display.clear()  # Displayet nullstilles

    sleep(300)  # Løkka settes på pause i 300 millisekunder