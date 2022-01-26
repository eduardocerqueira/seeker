#date: 2022-01-26T17:03:50Z
#url: https://api.github.com/gists/5823d3762a494458bddcabcfef61c2ce
#owner: https://api.github.com/users/overas

# Write your code here :-)
# Radiomottaker til å ha utenfor Faradaybur

from microbit import *
import radio

radio.on()  # Radio skrus på
radio.config(channel=42)  # Radiokanal settes til kanal 42

while True:  # Løkke som kjører til den blir avbrutt

    if radio.receive() is not None:  # Hvis den mottar et radiosignal skjer følgende:

        display.show(Image.HAPPY)  # Et smilefjes vises på displayet
    sleep(50)  # Løkka settes på pause i 50 millisekunder

    display.clear()  # Displayet tømmes
