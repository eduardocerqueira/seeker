#date: 2024-03-07T18:33:36Z
#url: https://api.github.com/gists/12838f2631bcb076e52072400149604a
#owner: https://api.github.com/users/eitafeio

#Caso de Recursão - Fatorial

def fatorial(n):
    if n == 1:
        return 1
    else:
        return n * fatorial(n - 1)

while True:
    n = int(input("Fatorial de 1 até: "))
    print("Resultado: ",fatorial(n) )