#date: 2023-05-31T16:58:55Z
#url: https://api.github.com/gists/e853397ac9017995568397b8772f07e8
#owner: https://api.github.com/users/ParkerMeyers

from math import sin,cos,pi
from random import randint
from kandinsky import *
from ion import *
from time import sleep
colors=[color(0,0,255),color(50,255,50),color(255,0,255),color(255,120,0),color(180,50,50)]
amogus=[[40,40,color(0,0,255)],[10,10,color(50,255,50)],[100,40,color(255,0,255)],[80,35,color(255,120,0)],[45,35,color(180,50,50)]]
amogus2=[]
amogusx=0
amogusy=0
black=color(0,0,0)
red=color(255,0,0)
gray100=color(100,100,100)
gray80=color(80,80,80)
white=color(255,255,255)
def movemogus(amogus,x,y):
  amogusx=amogus[j][0]+x
  amogusy=amogus[j][1]+y    
  for i in range(15):
    if get_pixel(amogus[j][0]+(x*i),amogus[j][1]+(x*i))==black:
      amogusx=amogus[j][0]
      amogusy=amogus[j][1]
      break
  return([amogusx,amogusy,amogus[j][2]])
      
def drawgun(frame):
  if frame==1:
    fill_rect(150,150,30,10,gray80)
    fill_rect(140,160,50,50,gray100)
    fill_rect(140,180,50,60,red)
  elif frame==2:
    fill_rect(170,175,70,10,gray80)
    fill_rect(220,180,30,35,gray100)
    fill_rect(230,190,30,50,red)
def drawmap(amogus):
  fill_rect(0,0,320,50,white)
  fill_rect(0,0,150,50,black)
  fill_rect(2,2,146,46,white)
  fill_rect(2,20,26,2,black)
  fill_rect(28,20,2,10,black)
  fill_rect(130,40,20,2,black)
  fill_rect(10,30,2,10,black)
  fill_rect(10,30,10,2,black)
  draw_string("HP: "+str(hp),245,18)
  draw_string("Score: "+str(score),205,0)
  for i in range(len(amogus)):
    fill_rect(amogus[i][0],amogus[i][1],2,2,amogus[i][2])
def render(amogus,x,y,direction,fov,view_dis):
  for i in range(16):
    drawmap(amogus)
    if keydown(KEY_SINE):
      drawgun(1)
    else:
      drawgun(2)
    draw_string(str(chamber)+"/6",0,186)
    draw_string("Total bullets: "+str(bullets),0,204)
    depth=view_dis
    for j in range(2,view_dis,1):
      if get_pixel((int(cos(pi*(direction+(i*fov))/180)*j))+x,(int(sin(pi*(direction+(i*fov))/180)*j))+y)==color(0,0,0):
        depth=j
        break
      fill_rect((int(cos(pi*(direction+(i*fov))/180)*j))+x,(int(sin(pi*(direction+(i*fov))/180)*j))+y,1,1,color(0,255,0))
    fill_rect(i*20,50,20,160,color(depth*int(255/view_dis),depth*int(255/view_dis),depth*int(255/view_dis)))
    fill_rect(i*20,50,20,depth*int(80/view_dis),color(50,180,255))
    fill_rect(i*20,222,20,(-1*(depth*int(80/view_dis)))-12,color(100,50,50))
  a=0
  for i in range(16):
    for j in range(2,view_dis,1):
      if get_pixel((int(cos(pi*(direction+(i*fov))/180)*j))+x,(int(sin(pi*(direction+(i*fov))/180)*j))+y)==color(0,0,0):
        break
      elif (get_pixel((int(cos(pi*(direction+(i*fov))/180)*j))+x,(int(sin(pi*(direction+(i*fov))/180)*j))+y)in colors):
        if (get_pixel(a*20,160)!=get_pixel((int(cos(pi*(direction+(i*fov))/180)*j))+x,(int(sin(pi*(direction+(i*fov))/180)*j))+y)):
          a=i
          fill_rect((i*20)-int(30*(view_dis-j)/view_dis),125+int(150/(2*view_dis)),int(60*(view_dis-j)/view_dis),int(70*(view_dis-j)/view_dis),get_pixel((int(cos(pi*(direction+(i*fov))/180)*j))+x,(int(sin(pi*(direction+(i*fov))/180)*j))+y))
          fill_rect((i*20)-int(30*(view_dis-j)/view_dis),125+int(150/(2*view_dis)),int(20*(view_dis-j)/view_dis),int(90*(view_dis-j)/view_dis),get_pixel((int(cos(pi*(direction+(i*fov))/180)*j))+x,(int(sin(pi*(direction+(i*fov))/180)*j))+y))
          fill_rect((i*20)+int(10*(view_dis-j)/view_dis),125+int(150/(2*view_dis)),int(20*(view_dis-j)/view_dis),int(90*(view_dis-j)/view_dis),get_pixel((int(cos(pi*(direction+(i*fov))/180)*j))+x,(int(sin(pi*(direction+(i*fov))/180)*j))+y))
          fill_rect((i*20)-int(25*(view_dis-j)/view_dis),130+int(150/(2*view_dis)),int(50*(view_dis-j)/view_dis),int(20*(view_dis-j)/view_dis),color(50,200,255))
hp=100          
fov=5
dirvel=0
direction=0    
x=20
y=40
view_dis=30
score=0
chamber=6
bullets=30
while True:
  if keydown(KEY_UP):
    if get_pixel(x+1,y-1)==black:
      y+=2
    else:
      y-=2
  elif keydown(KEY_DOWN):
    if get_pixel(x+1,y+3)==black:
      y-=2
    else:
      y+=2
  elif keydown(KEY_LEFT):
    if get_pixel(x-1,y+1)==black:
      x+=2
    else:
      x-=2
  elif keydown(KEY_RIGHT):
    if get_pixel(x+3,y+1)==black:
      x-=2
    else:
      x+=2
  if keydown(KEY_FOUR):
    dirvel-=10
  elif keydown(KEY_SIX):
    dirvel+=10
  else:
    dirvel=int(dirvel/4)
  direction+=dirvel
  render(amogus,x,y,direction,fov,view_dis)
  fill_rect(x,y,2,2,color(50,50,50))
  if keydown(KEY_TANGENT):
    if bullets>0:
      if bullets-(6-chamber) >=0:
        bullets-=6-chamber
        chamber+=6-chamber
      else:
        bullets-=6-chamber
        chamber+=6-chamber+bullets
        bullets-=bullets
  if keydown(KEY_SINE):
    if keydown(KEY_COSINE):
      if chamber!=0:
        chamber-=1
        for i in range(140,190,2):
          if get_pixel(i,135) in colors:
            draw_string("KILL +50",130,80,black,color(50,255,0))
            score+=50
            amogus2=[]
            for j in range(len(amogus)):
              if get_pixel(i,135)in amogus[j]:
                amogusx=randint(0,148)
                amogusy=randint(0,48)
                while get_pixel(amogusx,amogusy)==black:
                  amogusx=randint(0,150)
                  amogusy=randint(0,50)
                amogus2.append([amogusx,amogusy,get_pixel(i,135)])
              else:
                amogus2.append(amogus[j])
            amogus=amogus2
            break
        fill_rect(140,140,50,50,color(255,180,50))
      else:
        draw_string("Reload!",130,80)
    drawgun(1)
  else:
    drawgun(2)
  draw_string(str(chamber)+"/6",0,186)
  draw_string("Total bullets: "+str(bullets),0,204)
  amogus2=[]            
  for j in range(len(amogus)):
    if amogus[j][1]-y<15 and amogus[j][1]-y>3: #down
      if amogus[j][0]-x<15 and amogus[j][0]-x>3:#right
        amogus2.append(movemogus(amogus,-1,-1))
      elif x-amogus[j][0]<15 and x-amogus[j][0]>3:#left
        amogus2.append(movemogus(amogus,1,-1))
      elif x-amogus[j][0]<=3 and amogus[j][0]-x>=-3: #xcenter
        amogus2.append(movemogus(amogus,0,-1))
      else:
        amogus2.append([amogus[j][0],amogus[j][1],amogus[j][2]])
    elif y-amogus[j][1]<15 and y-amogus[j][1]>3: #up
      if amogus[j][0]-x<15 and amogus[j][0]-x>3:#right
        amogus2.append(movemogus(amogus,-1,1))
      elif x-amogus[j][0]<15 and x-amogus[j][0]>3:#left
        amogus2.append(movemogus(amogus,1,1))
      elif x-amogus[j][0]<=3 and amogus[j][0]-x>=-3: #xcenter
        amogus2.append(movemogus(amogus,0,1))
      else:
        amogus2.append([amogus[j][0],amogus[j][1],amogus[j][2]])
    elif amogus[j][1]-y<=3 and y-amogus[j][1]>=-3: #ycenter
      if amogus[j][0]-x<15 and amogus[j][0]-x>3:#right
        amogus2.append(movemogus(amogus,-1,0))
      elif x-amogus[j][0]<15 and x-amogus[j][0]>3:#left
        amogus2.append(movemogus(amogus,1,0))
      elif amogus[j][0]-x<=3 and x-amogus[j][0]>=-3: #xcenter
        hp-=2
        draw_string("Ouch!",140,80,black,red)
        amogus2.append([amogus[j][0],amogus[j][1],amogus[j][2]])
      else:
        amogus2.append([amogus[j][0],amogus[j][1],amogus[j][2]])
    else:
      amogus2.append([amogus[j][0],amogus[j][1],amogus[j][2]])
  if hp<=-1000:
    draw_string("You died",125,80)
    break
  amogus=amogus2
  draw_string(str(amogus[0][0]),150,100)
  sleep(0.2)
