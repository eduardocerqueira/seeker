#date: 2022-07-01T17:08:06Z
#url: https://api.github.com/gists/c7a3f07ec28c41163033b465b0893f8e
#owner: https://api.github.com/users/TheBonnec

_C=" Saison 2 : l'automne du monde "
_B=' -war just beggin...- '
_A=' -Wood Ungear Liquid 3- '
from math import *
from kandinsky import *
from ion import *
from time import *
from random import *
posxz=[]
posyz=[]
pjk=[3]
degmob=[1]
ml=[]
m='m'
for i in range(200):ml.append(m+str(i))
for i in range(33):posxz.append(i*10)
for i in range(22):posyz.append(i*10)
k=keydown
fl=fill_rect
dr=draw_string
clz={0:color(0,255,0),1:color(150,254,138),2:color(200,254,58),3:color(200,50,0),7:color(255,255,0)}
class dech:
  def __init__(self,x,y,c):self.x=x;self.y=y;self.c=c
  def rendu(self):fl(self.x-1,self.y-1,22,23,(0,0,0));fl(self.x,self.y,20,20,clz[self.c]);fl(self.x+3,self.y+3,5,5,(0,0,0));fl(self.x+13,self.y+3,5,5,(0,0,0));fl(self.x+4,self.y+4,3,3,(255,0,0));fl(self.x+14,self.y+4,3,3,(255,0,0));fl(self.x-1,self.y+20,22,52,(0,0,0));fl(self.x,self.y+23,20,50,clz[self.c]);fl(self.x-11,self.y+20,11,82,(0,0,0));fl(self.x-10,self.y+21,9,80,clz[self.c]);fl(self.x+20,self.y+20,11,82,(0,0,0));fl(self.x+21,self.y+21,9,80,clz[self.c]);fl(self.x-1,self.y+72,11,40,(0,0,0));fl(self.x,self.y+71,9,40,clz[self.c]);fl(self.x+9,self.y+72,12,40,(0,0,0));fl(self.x+10,self.y+71,10,40,clz[self.c]);fl(self.x-1,self.y+21,22,18,clz[self.c]);fl(self.x-6,self.y+95,4,4,(0,0,0));fl(self.x+22,self.y+95,4,4,(0,0,0));set_pixel(self.x-10,self.y+20,(66,66,66));set_pixel(self.x-11,self.y+20,(66,66,66));set_pixel(self.x-11,self.y+21,(66,66,66));set_pixel(self.x+29,self.y+20,(66,66,66));set_pixel(self.x+30,self.y+20,(66,66,66));set_pixel(self.x+30,self.y+21,(66,66,66))
class Joueur:
  def __init__(self,x,y,hp,d):self.d=d;self.x=x;self.y=y;self.hp=hp
  def move(self):
    if k(KEY_UP):
      self.d=3
      if self.y!=0:self.y-=10
    if k(KEY_RIGHT):
      self.d=1
      if self.x!=310:self.x+=10
    if k(KEY_DOWN):
      self.d=4
      if self.y!=210:self.y+=10
    if k(KEY_LEFT):
      self.d=2
      if self.x!=0:self.x-=10
  def rendu(self,collor,collor2):fl(self.x-1,self.y-1,12,12,collor2);fl(self.x,self.y,10,10,(255,0,0));dr('HP : '+str(self.hp),0,0,(250,100,0),collor)
  def globall(self,collor,collor2):self.move();self.rendu(collor,collor2)
class Mob:
  def __init__(self,x,y,c,j,dis,dellz):self.dellz=dellz;self.dis=dis;self.j=j;self.x=x;self.y=y;self.c=c
  def move(self):
    if self.dis==1 and self.dellz==pjk[0]:
      if self.x<self.j.x:self.x+=10
      elif self.x>self.j.x:self.x-=10
      elif self.y<self.j.y:self.y+=10
      elif self.y>self.j.y:self.y-=10
      self.dellz=0
    elif self.dellz<pjk[0]and pjk[0]!=0:self.dellz+=1
    if self.dellz>pjk[0]and pjk[0]!=0:self.dellz=0
  def globall(self):
    if self.dis==1:
      self.move()
      if self.x==self.j.x and self.j.y==self.y:self.j.hp-=degmob[0]
  def rendu(self):
    if self.dis==1:fl(self.x-1,self.y-1,12,12,(0,0,255));fl(self.x,self.y,10,10,self.c)
class tir:
  def __init__(self,x,y,d,dis,j,nn,nnn):self.nnn=nnn;self.nn=nn;self.j=j;self.dis=dis;self.x=x;self.y=y;self.d=d
  def move(self):
    if self.dis==1:
      if self.d==1:self.x+=10
      if self.d==2:self.x-=10
      if self.d==3:self.y-=10
      if self.d==4:self.y+=10
  def tir(self):
    if k(KEY_OK)and self.dis==0:self.x=self.j.x;self.y=self.j.y;self.d=self.j.d;self.dis=1
  def rendu(self,collor):fl(self.x+2,self.y+2,6,6,collor)
  def colider(self):
    if self.x<0 or self.x>310 or self.y<0 or self.y>210:self.dis=0;self.dis=0
  def collids(self,m):
    if self.dis==1:
      if self.x<0 or self.x>310 or self.y<0 or self.y>210:self.dis=0;self.dis=0
      if self.x==m.x and self.y==m.y and m.dis==1:m.dis=0;self.nnn+=1;self.nn+=1;self.dis=0;self.dis=0
  def globall(self):self.tir();self.collids();self.rendu((255,255,0))
def mainn():
  E='...';D=True;C='oui';B='non';A=False;fill_rect(0,0,333,333,(0,0,0));sleep(0.5);dr(_A,40,30,(255,255,0),(0,0,0));dr(_B,50,50,(255,255,0),(0,0,0));dr(_C,0,80,(250,100,50),(0,0,0));intoo=B
  while not k(KEY_OK):
    dr(' intro : < '+str(intoo)+' >',85,150,(255,255,0),(0,0,0))
    if k(KEY_LEFT):intoo=B
    if k(KEY_RIGHT):intoo=C
  j1=Joueur(100,100,100,3);nochia=0;t1=tir(-20,-20,j1.d,0,j1,nochia,0);running=D;intro=A;nbz=1
  def spawner(nbz,j):
    for i in range(nbz):az=choice(posxz);bz=choice(posyz);c=randint(0,3);ml[i]=Mob(az,bz,clz[c],j1,1,3)
  spawner(nbz,j1)
  if intoo==C:intro=D;o1=Mob(200,200,clz[1],j1,1,3)
  timel=0;olo=0;arte=0
  while intro:
    fl(0,0,333,333,(66,66,66));timel+=1
    if k(KEY_ZERO):intro=A
    t1.tir()
    for i in range(3):t1.move()
    if arte==2:t1.collids(o1)
    else:t1.move();t1.colider()
    if t1.dis==1 and arte==0:t1.rendu((0,0,0))
    elif t1.dis==1 and arte==2:t1.rendu((255,255,0))
    if arte==2:j1.globall((66,66,66),(255,255,0))
    else:j1.globall((66,66,66),(0,0,0))
    o1.globall();o1.rendu();sleep(0.05);j1.hp=100
    if timel==80:
      arte=1
      def passs():
        while not k(KEY_OK):0
      iii1=dech(10,140,7);iii1.rendu();fl(1,200,329,20,(255,255,255));fl(4,189,22,14,(0,0,0));fl(5,190,20,12,(255,255,255));dr(E,4,187,(0,0,0),(255,255,255),1);dr(' je vois que tes balles ne marchent pas...  ',0,203,(255,0,0),(255,255,255),1);sleep(1);passs();dr(' je vais etre grand, et te donner mon pouvoir.       ',0,203,(255,0,0),(255,255,255),1);sleep(1);passs();dr('au fait,ce que tu vois est un decharne               ',0,203,(255,0,0),(255,255,255),1);sleep(1);passs();dr('va au centre, et acquiert ma puissance             ',0,203,(255,0,0),(255,255,255),1);sleep(1);passs()
    if o1.dis==0:
      def passs():
        while not k(KEY_OK):0
      sleep(1);iii1.rendu();fl(1,200,329,20,(255,255,255));fl(4,189,22,14,(0,0,0));fl(5,190,20,12,(255,255,255));dr(E,4,187,(0,0,0),(255,255,255),1);dr(' tu a battu ce decharne, bravo!',0,203,(255,0,0),(255,255,255),1);sleep(1);passs();dr(' mais a present, tu dois savoir...          ',0,203,(255,0,0),(255,255,255),1);sleep(1);passs();dr(" il y en a plein d'autre, maintenant        ",0,203,(255,0,0),(255,255,255),1);sleep(1);passs();dr(' si tu a pu acquerir ma puissance...       ',0,203,(255,0,0),(255,255,255),1);sleep(1);passs();dr(" il y'a une raison                       ",0,203,(255,0,0),(255,255,255),1);sleep(1);passs();dr(' Idari. tu est toi meme un decharne.       ',0,203,(255,0,0),(255,255,255),1);sleep(1);passs();dr(' celui qui nous sauvera. ou nous detruira ',0,203,(255,0,0),(255,255,255),1);sleep(1);passs();dr(' va, a present.                           ',0,203,(255,0,0),(255,255,255),1);sleep(1);passs();intro=A
    if timel>80 and arte==1:
      fill_rect(99,99,12,12,(255,255,0));sleep(0.03)
      if j1.x==100 and j1.y==100:arte=2
  while running:
    if k(KEY_SHIFT):
      sleep(1)
      while not k(KEY_SHIFT):
        pass
      if k(KEY_SHIFT):
        sleep(1)
    if k(KEY_ZERO):running=A
    for i in range(nbz):ml[i].globall()
    fl(0,0,333,333,(0,0,0));dr(' score : '+str(t1.nnn),200,0,(250,100,0),(0,0,0));t1.tir()
    for i in range(3):
      t1.move()
      for i in range(nbz):t1.collids(ml[i])
    if t1.dis==1:t1.rendu((255,255,0))
    j1.globall((0,0,0),(255,255,0))
    for i in range(nbz):ml[i].rendu()
    sleep(0.05)
    if t1.nn==nbz:nbz+=1;t1.nn=0;spawner(nbz,j1)
    if j1.hp<=0:fl(0,0,333,333,(0,0,0));dr(' les decharnes ont eus raisons de vous...',0,50,(255,100,0),(0,0,0),1);dr('score : '+str(t1.nnn),100,70,(250,100,0),(0,0,0));sleep(3);running=A
    if t1.nnn>=150:pjk[0]=2
    if t1.nnn>=500:pjk[0]=1
    if t1.nnn>=200:degmob[0]=1.5
    if t1.nnn>=500:degmob[0]=2
    if t1.nnn>=800:degmob[0]=2.5
    if t1.nnn>=1000:degmob[0]=3
    if t1.nnn>=1200:degmob[0]=3.5
    if t1.nnn>=1500:degmob[0]=4
    if t1.nnn>=1800:degmob[0]=4.5
    if t1.nnn>=2000:degmob[0]=5
def launch():fill_rect(0,0,333,333,(0,0,0));sleep(0.5);fill_rect(100,50,100,10,(255,0,0));fill_rect(100,50,10,100,(255,0,0));fill_rect(100,100,100,10,(255,0,0));fill_rect(100,150,100,10,(255,0,0));fill_rect(140,50,10,100,(255,0,0));draw_string("+++ Ersatz d'Eratz studio +++",50,160,(255,0,0),(0,0,0),1);sleep(2);fill_rect(0,0,333,333,(0,0,0));sleep(0.5);dr(_A,40,30,(255,255,0),(0,0,0));dr(_B,50,50,(255,255,0),(0,0,0));dr(_C,0,80,(250,100,0),(0,0,0));sleep(2);mainn()# This comment was added automatically to allow this file to save.
# You'll be able to remove it after adding text to the file.
