#date: 2024-02-09T17:03:44Z
#url: https://api.github.com/gists/ca340717e1fa765d2b379aa734b469a0
#owner: https://api.github.com/users/m-bchr

from kandinsky import fill_rect as rct,get_pixel as pxl,draw_string as dstr
from ion import *
from time import sleep
from random import randint
PNK,DGBLU,RED,DBRW,BRW=(255,200,150),(20,130,150),(255,50,50),(120,50,0),(200,100,50)
OWHT,GBLU,PUR,BWHT,BLK=(240,)*3,(29,207,207),(240,240,255),(205,225,245),(0,)*3
PBLU,YLO,BLU,WHT=(150,150,255),(255,180,50),(50,50,255),(255,)*3

def draw_player(x,y,d):
  rct(x-4*d,y,8*d,-2,YLO)
  rct(x-2*d,y-2,-4*d,-10,PBLU)
  rct(x-4*d,y-12,10*d,-4,PBLU)
  rct(x-2*d,y-2,8*d,-12,WHT)
  rct(x-4*d,y-4,4*d,-8,BLU)
  rct(x+4*d,y-12,4*d,2,YLO)
  rct(x+4*d,y-12,-2*d,-2,BLK)
def draw_water(sx,sy):
  rct(0,100+sy,320,250,GBLU)
  rct(0,350+sy,320,300,DGBLU)
def draw_back(sx,sy,watery):
  rct(0,-120+sy,320,120,BWHT)
  rct(0,0+sy,320,watery-sy,PUR)
def draw_ground(sx,sy):
  rct(-500+sx,-200+sy,20,800,WHT)
  rct(500+sx,-200+sy,20,800,WHT)
  rct(-500+sx,-200+sy,1000,20,WHT)
  rct(-350+sx,380+sy,750,20,WHT)
  rct(400+sx,380+sy,20,100,WHT)
  rct(-100+sx,500+sy,350,20,WHT)
  rct(sx,500+sy,20,100,WHT)
  rct(-150+sx,400+sy,20,100,WHT)
  rct(-350+sx,450+sy,20,150,WHT)
  rct(-500+sx,600+sy,1020,20,WHT)
  rct(-450+sx,100+sy,280,-20,WHT)
  rct(-250+sx,100+sy,20,-100,WHT)
  rct(-250+sx,-50+sy,20,-150,WHT)
  rct(150+sx,100+sy,200,-20,WHT)
  rct(350+sx,100+sy,20,-100,WHT)
def draw_fish(x,y,d,type):
  if type==1:
    rct(x,y,10*d,10,BWHT)
    rct(x,y+3,-4*d,4,PBLU)
    rct(x-4*d,y,-4*d,10,BWHT)
    rct(x+2*d,y,4,-2,PBLU)
    rct(x+6*d,y+2,2*d,2,BLK)
  elif type==3:
    rct(x-10*d,y+4,36*d,4,PBLU)
    rct(x,y,30*d,6,BWHT)
    rct(x+4*d,y+2,2*d,2,BLK)
    rct(x+6*d,y,-10*d,-2,PBLU)
    rct(x-10*d,y,-4*d,8,BWHT)
  elif type==0:
    rct(x-12*d,y,25*d,15,PNK)
    rct(x-5*d,y+5,10*d,7,RED)
    rct(x-7*d,y+15,5*d,15,PNK)
    rct(x+2*d,y+15,5*d,15,PNK)
def draw_orca(x,y,d):
  rct(x,y,36*d,24,BLK)
  rct(x,y+24,30*d,6,OWHT)
  rct(x,y+9,-12*d,12,OWHT)
  rct(x-12*d,y,-12*d,30,BLK)
  rct(x+6*d,y,-12*d,-6,BLK)
  rct(x+15*d,y+6,9*d,6,OWHT)
def draw_box(x,y):
  rct(x,y+6,20,14,DBRW)
  rct(x,y,20,6,BRW)
  rct(x+8,y+4,4,6,OWHT)
def draw_squid(x,y):
  rct(x-60,y-40,200,80,RED)
  rct(x+30,y-15,20,20,BLK)
  rct(x+60,y-30,80,60,DGBLU)
def draw_letext(score,invinc,life,infair,air,hunger):
  rct(0,0,320,18,WHT)
  dstr("Pts:"+str(score),0,0)
  if invinc>0: dstr("HP:"+str(int(life)),100,0,YLO)
  elif life<5: dstr("HP:"+str(int(life)),100,0,RED)
  else: dstr("HP:"+str(int(life)),100,0)
  if infair>0: dstr("Air:"+str(int(air)),160,0,YLO)
  elif air<20: dstr("Air:"+str(int(air)),160,0,RED)
  else: dstr("Air:"+str(int(air)),160,0)
  dstr("Hunger:"+str(hunger),230,0,RED*(hunger<5) or BLK)
def game():
  sx=0
  sy=100
  yvel=10
  dir=1
  xvel=0
  score=0
  time=0
  fish=[]
  fish2=[]
  box=[]
  box2=[]
  ox=0
  oy=200
  odir=1
  a=1
  hunger=10
  life=10
  invinc=20
  boost=0
  air=40
  infair=0
  squidx=500
  squidy=0
  while True:
    if time%110==0: fish.append([randint(-480,480),randint(400,550),randint(1,2)*2-3,0])
    if time%70==0: fish.append([randint(-480,480),randint(100,350),randint(1,2)*2-3,1])
    if time%150==0: fish.append([randint(-480,480),randint(300,350),randint(1,2)*2-3,3])
    if time%70==69: hunger-=1
    if time%100==0:
      life=int(life)
      life+=1
      if life>10: life=10
    if time%300==0:
      squidx=-500
      squidy=randint(400,600)
    if time%500==0: box.append([randint(-450,-300),60])
    draw_back(int(sx/2),int(sy/2),100+sy)
    draw_water(sx,sy)
    draw_player(160,110,dir*(int(invinc)%2))
    draw_orca(ox+sx,oy+sy,odir)
    if squidx<500:
      draw_squid(squidx+sx,squidy+sy)
      squidx+=15
      if abs(sx+squidx-160)<80 and abs(sy+squidy-98)<50:
        if invinc<0: life-=0.5
        if life<=0:
          dstr("The squid ate you!",80,100)
          break
    for i in range(len(fish)): draw_fish(fish[i][0]+sx,fish[i][1]+sy,fish[i][2],fish[i][3])
    draw_ground(sx,sy)
    fish2=[]
    for i in range(len(fish)):
      if fish[i][0]<=-450 or fish[i][0]>=450: a=-1
      else: a=1
      if abs(sx+fish[i][0]-160)<25 and abs(sy+fish[i][1]-98)<20:
        if fish[i][3]==0: score+=40
        else: score+=10*fish[i][3]
        if hunger<10: hunger+=1
      else: fish2.append([fish[i][0]+(3*fish[i][2]*a*fish[i][3]),fish[i][1],fish[i][2]*a,fish[i][3]])
    fish=fish2
    box2=[]
    for i in range(len(box)):
      if abs(sx+box[i][0]-160)<20 and abs(sy+box[i][1]-98)<20:
        a=randint(1,5)
        if a==1: score+=50
        elif a==2:
          life=int(life)
          life+=3
          if life>10: life=10
        elif a==3: invinc=30
        elif a==4: boost=200
        elif a==5: infair=200
      else: box2.append([box[i][0],box[i][1]])
    box=box2
    if oy<330 and sy+oy-98<0: oy+=6
    elif oy>100 and sy+oy-98>0: oy-=6
    if abs(sx+ox-160)<200 and abs(sy+oy-98)<100 and oy>100:
      if sx+ox-160<0:
        ox+=7
        odir=1
      elif sx+ox-160>0:
        ox-=7
        odir=-1
    else: ox+=odir*5
    if ox<=-450: odir=1
    elif ox>=450: odir=-1
    for i in range(len(box)): draw_box(box[i][0]+sx,box[i][1]+sy)
    draw_letext(score,invinc,life,infair,air,hunger)
    if sy<0:
      if infair<=0: air-=0.1
      if air<=0:
        air=0
        if invinc<0: life-=0.2
        if life<=0:
          dstr("You died of drowning!",60,100)
          break
    elif air<40:
      air=int(air)
      air+=1
    if abs(sx+ox-160)<40 and abs(sy+oy-90)<30:
      if invinc<0: life-=0.2
      if life<=0:
        dstr("The orca ate you!",80,100)
        break
    if hunger==0:
      if invinc<0: life-=0.2
      if life<=0:
        dstr("You died of hunger!",70,100)
        break
    if keydown(0)and yvel<1:
      dir=-1
      yvel=10
      if boost>1: xvel=10
      else: xvel=5
    elif keydown(3)and yvel<1:
      dir=1
      yvel=10
      if boost>1: xvel=-10
      else: xvel=-5
    if pxl(160,95)==WHT:
      yvel=-2
      sy-=10
    if pxl(145,105)==WHT and dir==-1: xvel=0
    elif pxl(170,105)==WHT and dir==1: xvel=0
    sx+=xvel
    sy+=yvel
    if pxl(160,110)==WHT:
      yvel=0
      xvel=int(xvel/1.2)
      while pxl(160,110)==WHT:
        sy+=1
        draw_back(int(sx/2),int(sy/2),100+sy)
        draw_water(sx,sy)
        draw_player(160,110,dir*(int(invinc)%2))
        draw_orca(ox+sx,oy+sy,odir)
        if squidx<500: draw_squid(squidx+sx,squidy+sy)
        for i in range(len(fish)): draw_fish(fish[i][0]+sx,fish[i][1]+sy,fish[i][2],fish[i][3])
        draw_ground(sx,sy)
        for i in range(len(box)): draw_box(box[i][0]+sx,box[i][1]+sy)
        draw_letext(score,invinc,life,infair,air,hunger)
    elif yvel>-7: yvel-=1
    time+=1
    if infair>-1: infair-=1
    if boost>-1: boost-=1
    if invinc>-1: invinc-=0.2
    sleep(0.03)

draw_back(0,20,100)
draw_water(0,0)
draw_player(180,140,1)
draw_fish(280,130,1,1)
draw_orca(50,110,1)
dstr("PENGUIN LIFE",100,30)
while True:
  if keydown(4): game()
  dstr("Press [OK] to play",75,150)
