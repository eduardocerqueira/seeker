#date: 2023-04-28T16:53:53Z
#url: https://api.github.com/gists/eb321880f22f3f172c56428624998640
#owner: https://api.github.com/users/SergioCuastumal

a= float(input("Nota 1: "))
b= float(input("Nota 2: "))
c= float(input("Nota 3: "))
def nn(a,b,c)->float:     
    if ((a>b)and(b>c)):
        s=(a+b)/2
    else:
        s=(a+c)/2
    if ((a>c)and(c>b)):
        s=(a+c)/2
    else:
        s=(b+c)/2
    return s
print (nn( a,b,c))
## Se agregaron operadores lógicos para los condicionales y la función que retorna el promedio