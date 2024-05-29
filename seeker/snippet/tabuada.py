#date: 2024-05-29T17:10:59Z
#url: https://api.github.com/gists/b0b0e291ccd6462b9cb888d594f13e78
#owner: https://api.github.com/users/Tg7vg

while True:
    try:
        x = int(input("Qual número você deseja saber a tabuada? "))
        break
    except ValueError:
        print("Somente números inteiros.")

while True:
    try:
        y = int(input("Você deseja que vá até onde? "))
        if y < 0:
            print("Somente valores inteiros positivos.")
        else:
            break
    except ValueError:
        print("Defina um número inteiro.")

r = 0
p = 0

def f():
    print(f"{x} x {p} = {x*p}")
print("____________________________")
while r <= y:
    f()
    r = r + 1
    p = p + 1
print("____________________________")
