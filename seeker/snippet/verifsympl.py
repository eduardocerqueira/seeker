#date: 2021-12-17T16:44:32Z
#url: https://api.github.com/gists/7f7bcde5fb3992e4db04f5985ab4fb65
#owner: https://api.github.com/users/Lucas301411

from math import *
from time import *
from kandinsky import *
from time import *



##MENU##
print("\n\n\n")
print("MENU:\n1.Distance entre 2 points\n2.Milieu\n3.Perimetre cercle\n4.Pythagore\n5.Coordonee vecteur\n6.Pourcentage")
choix=int(input("Choix: "))
########

#DISTANCE ENTRE DEUX POINT
if choix==1:
print("\n\n\n\n\n\n\n\n\n")
print("1.Distance entre deux points")
print("A(xa;ya) B(xb;yb)")
xa=float(input("XA:"))
ya=float(input("YA:"))
xb=float(input("XB:"))
yb=float(input("YB:"))
dist=sqrt((xb-xa)**2+(yb-ya)**2)
print("Distance AB:",round(dist,3))

#MILIEU
if choix==2:
print("\n\n\n\n\n\n\n\n\n")
print("2.Milieu")
print("A(xa;ya) B(xb;yb)")
xa=float(input("XA:"))
ya=float(input("YA:"))
xb=float(input("XB:"))
yb=float(input("YB:"))
xi=(xa+xb)/2
yi=(ya+yb)/2
print("Coordonees milieu:I(",xi,";",yi,")")

#PERIMETRE CERCLE
if choix==3:
print("\n\n\n\n\n\n\n\n\n\n")
print("3.Perimetre cercle")
r=float(input("Rayon:"))
peri=2*pi*r
print("Perimetre cercle:",peri)

#PYTHAGORE
if choix==4:
print("\n\n\n\n\n\n\n\n\n\n")
print("4.Pythagore")
ab=float(input("AB:"))
ac=float(input("AC:"))
bc=sqrt(ab**2+ac**2)
print("Hypotenuse:",bc)

#VECTEUR
if choix==5:
print("\n\n\n\n\n\n\n\n\n\n")
print("5.Coordonee vecteur")
xa=float(input("XA:"))
ya=float(input("YA:"))
xb=float(input("XB:"))
yb=float(input("YB:"))
vecabx=xb-xa
vecaby=yb-ya
print("Vecteur AB (",vecabx,";",vecaby,")")

#POURCENTAGE
if choix==6:
print("\n\n\n\n\n\n\n\n\n\n")
print("6.Pourcentage")
a=float(input("Quantitee: "))
b=float(input("Total: "))
c=(a/b)*100
print("Le pourcentage est de ",c,"%")

#HELP
if choix==999:
print("\n\n\n\n\n\n\n\n")
print("&.Credits")
print("Cree par: lucas30")
print("Version 1.2")
#END
##INTRO###
fill_rect(0,0,320,222,color(50,20,75))
fill_rect(5,5,310,212,color(255,255,255))
def z(x0,y0,r,c1,e,c2):
for i in range(e):
…    print("\n\n\n\n\n\n\n\n\n\n")
print("5.Coordonee vecteur")
xa=float(input("XA:"))
ya=float(input("YA:"))
xb=float(input("XB:"))
yb=float(input("YB:"))
vecabx=xb-xa
