#date: 2022-01-13T17:04:18Z
#url: https://api.github.com/gists/b5125bf6246bcb9ea2af3a3164bd4887
#owner: https://api.github.com/users/Kodzila1

C = 33.8 #1 Celsius = 33.8 fahrenheits
celsius = float(input("How many Celsius do you want to convert? "))
if celsius== 12.5 or celsius== 10.3:
    print("so far")
elif celsius == 80.1:
    print("not my fault")
elif celsius!=12.5 and celsius!=10.3 and celsius!=80.1:
    print(celsius*C)
else:
    print("please enter a valide number")

