#date: 2022-11-18T17:03:27Z
#url: https://api.github.com/gists/fe97c8474bc205931328683a9a664daa
#owner: https://api.github.com/users/ibrataha8

def Somme(n):
    if n==0:
        return 0
    else:
        return n + Somme(n-1)
a = int(input("Donner un nbre : "))
print('La somme est :',Somme(a))
