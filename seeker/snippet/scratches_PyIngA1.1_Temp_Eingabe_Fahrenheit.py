#date: 2022-02-18T16:54:01Z
#url: https://api.github.com/gists/31c1c293d6ceb4a0cabfce5ca942b597
#owner: https://api.github.com/users/HelloRamo

print("Umrechnung der Temperaturen von Celsius in Fahrenheit")
print("-----------------------------------------------------")
print()
Celsius = input("Bitte geben Sie eine Temperatur in Celsius ein: ")
Celsius = float(Celsius)

Fahrenheit = 9/5 * Celsius + 32
print()
print("Sie haben %f Grad Celsius eingegeben." % Celsius)
print()
print("Diese Temperatur entspricht %f Grad Fahrenheit." % Fahrenheit)