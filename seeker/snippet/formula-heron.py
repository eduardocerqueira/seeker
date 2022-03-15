#date: 2022-03-15T16:45:16Z
#url: https://api.github.com/gists/91f549bb87faf85971cda7db166c3fa5
#owner: https://api.github.com/users/dndsza

from math import sqrt


def calcula_area(a, b, c):
    try:
        s = 0.5 * (a + b + c)
        res = sqrt(s * (s - a) * (s - b) * (s - c))
        res = round(res, 2)
        return f"A área desse triângulo é {res} metros quadrados"
    except ValueError:
        return "Error!!"


def main():
    print("")
    print("Fórmula de Héron")
    try:
        a = int(input("Primeiro Lado: "))
        b = int(input("Segundo Lado: "))
        c = int(input("Terceiro Lado: "))
        print(calcula_area(a, b, c))
    except:
        print("Apenas inteiros/erro")


if __name__ == "__main__":
    main()
