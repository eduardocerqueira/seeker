#date: 2022-01-26T17:06:06Z
#url: https://api.github.com/gists/dac117812081d97b0964b99a28fb149c
#owner: https://api.github.com/users/overas

# Write your code here :-)
# Radiosender til å sette i Faradaybur

from microbit import *
import radio

radio.on()  # Radio skrus på

radio.config(
    channel=42, power=7
)  # Radiokanal settes til kanal 42 og signalstyrke på maksimal

display.show(Image.SAD)  # Et surfjes vises på displayet

while True:  # Løkke som kjører til den blir avbrutt
    radio.send("BEEP!")  # Radiosignalet 'BEEP' sendes
    sleep(1000)  # Løkka settes på pause i 1000 millisekunder
