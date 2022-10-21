#date: 2022-10-21T17:02:30Z
#url: https://api.github.com/gists/c8fc91ade9977bccdab7ffa31dc35a7c
#owner: https://api.github.com/users/ibrataha8

tab = []
N=int(input("Donner un nbr : "))
for i in range(0,N):
   tab.append(int(input('donner un nombre')))
n=int(input('Veuillez entrer la valeur de n : '))
x=0
for i in range(0,N):
   if n == tab[i]:
      x=x+1
if x==0 :
   print('n makynach')
else:
   print('n kayna')
