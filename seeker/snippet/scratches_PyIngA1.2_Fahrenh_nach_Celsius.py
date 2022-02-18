#date: 2022-02-18T16:54:01Z
#url: https://api.github.com/gists/31c1c293d6ceb4a0cabfce5ca942b597
#owner: https://api.github.com/users/HelloRamo

print("Umrechnung der Temperaturen von Fahrenheit in Celsius")
print("-----------------------------------------------------")
print()

Fahrenheit = input("Bitte geben Sie eine Temperatur in Fahrenheit ein: ")
Fahrenheit = float(Fahrenheit)
Celsius = 5 * (Fahrenheit - 32)/9

print()
print("Sie haben %f Grad Fahrenheit eingegeben." % Fahrenheit)
print()
print("Diese Temperatur entspricht %f Grad Celsius." % Celsius)