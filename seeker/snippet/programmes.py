#date: 2022-03-30T17:12:26Z
#url: https://api.github.com/gists/1b41835372f7a086f279849666ddd89f
#owner: https://api.github.com/users/kylianrcd

from math import *
from time import *
from kandinsky import *
from random import *
from ion import *

#--------------------------#

def multiple(n, limite):
  resultat = n
  while resultat < limite:
    resutat = resultat + abs(n)
  return resultat - abs(n)

#--------------------------#
    
def est_diviseur(a, b):
  if b%a==0:
    return True
  else:
    return False

#--------------------------#
      
def est_premier(n):
  for i in range(2,n):
    if est_diviseur(i,n):
      return False
  return True

#--------------------------#
  
def racine(r,n):
  a=1
  while a*a<r:
    a=a+10**-n
  return a-10**-n,a  

#--------------------------#
  
def e(n):
  a=1
  b=1
  i=1
  while i<n:
    c=sqrt(a*a+b*b)
    a=c
    i=i+1
  return(c)

#--------------------------#

def m(x1,y1,x2,y2):
  xm=(x1+x2)/2
  ym=(y1+y2)/2
  return (xm, ym)

#--------------------------#

def d(x1,y1,x2,y2):
  ab=sqrt((x2-x1)**2+(y2-y1)**2)
  return ab**2,ab

#--------------------------#

def fac(n):
  r=1
  while n > 0:
    r=r*n
    n=n-1
  return n
  
#--------------------------#

def pret(somme,remboursement):
  mois=0
  while somme>0:
    somme=somme-remboursement
    mois=mois+1
  return mois  
    
#--------------------------#

def ep():   #marche pas
  s=1
  l=1.6
  while l>=0.05:
    l=1.6*0.9
    s=s+1
  return s  
    
#--------------------------#

def distance_s(n):
  d=0
  for i in range(1,n+1):
    d=d+1.6*0.9**(i-1)
  return d  
    
#--------------------------#

def ep_2(x):
  s=1
  l=1.6
  while l>=x:
    l=l*0.9
    s=s+1
  return s  
    
#--------------------------#

#polynome 2nd degrés

def eq2(a,b,c):
  d=b**2-4*a*c
  if d<0:
    return d<0
    
  x1=(-b+sqrt(d))/2*a
  x2=(-b-sqrt(d))/2*a
  return x1,x2
  
#--------------------------#

#suite de collatz

def cz(u):
  L=[u]
  for i in range(1,50):
    import math
    if math.fmod(u,2)>0:
      u=3*u+1
      L.append(u)
    else:
      u=u/2
      L.append(u)
  return L    
  
#--------------------------#    

#test suite collazt  

def tsc(u):
  i=0
  while u!=1:
    import math
    if math.fmod(u,2)>0:
      u=3*u+1
      i=i+1
    else:
      u=u/2
      i=i+1
  return i    

#--------------------------#

#dessine un soleil mdr

def so():
  color('red', 'yellow')
  begin_fill()
  while True:
    forward(200)
    lef(170)
    if abs(pos())<1:
        break
    end_fill()
    done()
  
#--------------------------#

#Tron

FOND = (40, 48, 64)

class moto:
    dir = [[0,1],[1,0],[0,-1],[-1,0]]
    def __init__(self, nom, x, y, d, coul, ga, dr):
        self.nom = nom
        self.pos = [x, y]
        self.d = d
        self.coul = coul
        self.gauche = ga
        self.droite = dr
        self.tps = monotonic()
        self.fin=False
        
    def tourne(self, t):
      if monotonic() - self.tps > 0.2:
        self.tps = monotonic()
        self.d = (self.d + t) % 4

    def maj(self):
      if keydown(self.gauche): self.tourne(-1)
      if keydown(self.droite): self.tourne(1) 
      [x, y] = self.pos
      [dx, dy] = self.dir[self.d]
      self.pos = [x + dx, y + dy]
      [x, y] = self.pos
      if get_pixel(x, y) != FOND:
        self.fin=True
      else:  
        fill_rect(x,y,1,1,self.coul)

def go(n = 2):
  fill_rect(0,0,320,222,FOND)
  j1 = moto('ZNR',0,110,1,(255,255,0),KEY_ONE,KEY_SEVEN)
  j2 = moto('BARROUILLET',320,110,3,(255,255,255),KEY_RIGHTPARENTHESIS,KEY_MINUS)
  j3 = moto('PRAT',160,220,2,(0,255,0),KEY_ANS,KEY_DOT)
  j4 = moto('FAIBLE',160,0,0,(255,0,0),KEY_ALPHA,KEY_VAR)
  joueurs = [j1, j2]
  if n > 2: joueurs.append(j3)
  if n > 3: joueurs.append(j4)
  while len(joueurs) > 1:
    for j in joueurs:
     j.maj()
     if j.fin: joueurs.remove(j)
    sleep(.02)
  j = joueurs[0]
  draw_string(j.nom+" GAGNE !",100,200,j.coul, FOND)
  return j.nom
  

#--------------------------#

#liste de diviseurs

def dcz(n):
  ZR = []
  for i in range(1,n+1):
    if n%i==0:
      ZR.append(i)
  return ZR
  
#--------------------------#

#dm
    
def dm2():
  u=1
  S=u
  for k in range(1,100):
    u=u*exp(-u)
    S=S+u
  return S  

#--------------------------#

#exo oiseau

def piaf(l):
  o=600
  n=1
  while o<l:
    o=0.9*o+100
    n=n+1
  return n  

#--------------------------#

#activite pere noel

def al(n):
  L = [x + 1 for x in range(n)]
  L2 = [x+1 for x in range(n)]
  M = []
  M2 = []
  for i in range(n):
    c = choice(L)
    c2 = choice(L2)
    while c2==c:
      c=choice(L)
      c2=choice(L2)
    M.append(c)
    M2.append(c2)
    L.remove(c)
    L2.remove(c2)            
  return M, M2
  
#--------------------------#

#toutes les paires

#def paires(L):
  #for i in range(len(L)-2):
    #for j1 in range():
      
#--------------------------#

#ds suite

def suite(n):
  u=1000
  for i in range(n):
    u=0.9*u+250
  return u  

#--------------------------#    

#ds suite

def abonne():
  u=1000
  n=2020
  while u<2200:
    u=0.9*u+250
    n=n+1
  return n  

#--------------------------#

#planche de galton

def galton(n):
  c=[0,0,0,0,0,0,0,0,0,0,0,0,0]
  for i in range(n):
    d=0
    for j in range(12):
      d=d+randint(0,1)
    c[d]+=1
  return c    
      
#--------------------------#    
  
#pgcd

def pgcd(a,b):
  while a%b!=0:
    c=b  
    b=a%b
    a=c 
  return b
  
#--------------------------#

#pgcd_v2

def pgcd_v2(a,b):
  while a%b!=0:
    a,b=b,a%b
  return b  
    
#--------------------------#    

#calcul de pi par la méthode
#de Monte Carlo
        
def pimontecarlo(nHistoires):
  Compteur=0
  for i in range(0,nHistoires):
    x=random()
    y=random()
    if sqrt(x**2+y**2)<=1:
      Compteur = Compteur + 1
  print(4*Compteur/nHistoires)        
