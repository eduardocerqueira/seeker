#date: 2022-06-20T17:17:37Z
#url: https://api.github.com/gists/9376c9c4a54e4b802e458d1e8bb71eed
#owner: https://api.github.com/users/Frablock

from math import *
from math import *
from cmath import *
from turtle import *
from ion import *
from random import *
from kandinsky import *
from time import *
from os import *
goli="\n"*12
#gema
def gm_reset():
  reset()
  hideturtle()
def gm_error():
  print("err")
  gm_reset()
  draw_string("Une erreur est survenue",0,80)
def gm_clic(a,b,e):
  while True:
    if keydown(KEY_UP):
      return a
    if keydown(KEY_DOWN):
      if e>=2:
        return b
#fin gema
def h():
  try:
    rename("dataos.hpy","dataos.py")
  except :
    try:
      rename("dataos.py","dataos.hpy")
    except:
      print("erreur dataos")
#app
def app(a):
  print("starting : "+a)
  if a=="cmd":
    cmd()
  if a=="file":
    file()
  if a=="calc":
    calc()
def cmd():
  try:
    from cmd import *
  except :
    gm_error()
def file():
  gm_reset()
  draw_string(str(len(listdir()))+" fichiers\n"+str(listdir()),0,0)
  print(goli+str(listdir()))
  draw_string("[UP] Ouvrir un fichier\n[DOWN] Creer un fichier",0,80)
  if gm_clic(1,2,2)==2:
    try:
      open(input("Nom du fichier : "),"w")
      draw_string("Votre fichier a ete cree",0,80)
      sleep(1)
      file()
    except :
      gm_error()
      file()
  fi=str(showapp(listdir()))
  if fi=="ostest.py":
    print("Un fichier ne peut se modifier seul")
    file()
  sleep(0.25)
  a=showapp(["Lire","Editer"])
  if str(a)=="Lire":
    gm_reset()
    draw_string(open(fi).read(999999),0,0)
    gm_clic(1,0,1)
    sleep(0.5)
    file()
  elif str(a)=="Editer":
    print(open(fi).read(999999))
    txt=str(input())
    sleep(0.5)
    if showapp(["sauvegarder","non"])=="sauvegarder":
      open(fi,"w").write("##\n"+str(txt))
      draw_string("Fichier sauvegarde",0,80)
      sleep(1)
    file()
def calc():
  print(goli+"utilisez le clavier numerique")
#fin app
def shpp(a):
  la=len(a)
  c=["grey","white"]
  xd=0
  yd=0
  va=int(320/la)
  for i in range(la):
    fill_rect(xd,yd,va,225,c[i%2])
    draw_string(a[i],xd,0)
    xd=xd+va
def shmo(a,b):
  xd=int(320/len(a))*b+5
  fill_rect(xd,50,20,20,"red")
def showapp(a):
  p=0
  shpp(a)
  shmo(a,p)
  while True:
    if keydown(KEY_RIGHT):
      p=p+1
      if p>len(a):
        p=len(a)
      shpp(a)
      shmo(a,p)
      sleep(0.25)
    if keydown(KEY_LEFT):
      p=p-1
      if p<0:
        p=0
      shpp(a)
      shmo(a,p)
      sleep(0.25)
    if keydown(KEY_OK):
      sleep(0.25)
      app=a[p]
      return app
    if keydown(KEY_XNT):
      menu()
      sleep(0.25)
#start
def menu():
  print(goli)
  app(showapp(["cmd","file","calc","app","param"]))
try:
  h()
  from dataos import *
  l=logins
  h()
except :
  print("DATA introuvable")
x=l.index(input(goli+"utilisateur : "))
if divmod(x,2)[1]==0:
  if int(l.index(input("Mot de passe : ")))==int(x)+1:
    menu()
  else:
    print("Merci de redemarer ce script")
else:
  print("Merci de redemarer ce script")
