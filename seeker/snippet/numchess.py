#date: 2024-10-04T17:09:04Z
#url: https://api.github.com/gists/aa6260fe1b05d00f7c00a923eba0cf0e
#owner: https://api.github.com/users/GRocket-GamingGalaxy

from math import *
from kandinsky import *
from ion import *
from time import *

# Board
squareSize=24
gridX=10
gridY=10
gColor1=color(210,210,210)
gColor2=color(130,130,130)
playing=True
# FEN format
position="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
# Able to castle? (FEN)
castles="QKqk"
# Current players turn
player="w"
# Binary piece sprites
# Replace with your own (make resolution squareSize-2 by squareSize-2 if you don't want to fiddle with the code)
sprites={
  "king":"0000000000000000000000000000000011000000000000000000011110000000000000000001111000000000000000000011000000000000000000111111000000000000000111111110000000000000011111111000000000000001111111100000000000000011111100000000000000000111100000000000000000011110000000000000000001111000000000000000001111110000000000000000111111000000000000000011111100000000000000011111111000000000000001111111100000000000001111111111000000000001111111111110000000001111111111111100000000111111111111110000",
  "queen":"0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001100000000000000000001111000000000000000011111111000000000000011111111110000000000000111111110000000000000001111110000000000000000111111000000000000000001111000000000000000000111100000000000000000011110000000000000000001111000000000000000000111100000000000000000111111000000000000000011111100000000000000011111111000000000000011111111110000000000001111111111000000",
  "bishop":"0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011000000000000000000011110000000000000000011111000000000000000001111010000000000000001111111100000000000000111111110000000000000001111110000000000000000111111000000000000000001111000000000000000000111100000000000000000011110000000000000000001111000000000000000001111110000000000000001111111100000000000000111111110000000",
  "knight":"0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001111000000000000000011111110000000000000011111111100000000000001111111110000000000001111111111100000000000111111111110000000000001111110110000000000000111111100000000000000001111110000000000000000111111100000000000000011111111000000000000011111111110000000000001111111111000000",
  "rook":"0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000110110110000000000000011011011000000000000001111111100000000000000011111100000000000000001111110000000000000000111111000000000000000011111100000000000000001111110000000000000000111111000000000000000111111110000000000000011111111000000000000011111111110000000000001111111111000000",
  "pawn":"0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000111100000000000000000111111000000000000000011111100000000000000001111110000000000000000111111000000000000000001111000000000000000000111100000000000000000111111000000000000000011111100000000000000001111110000000000000001111111100000000000000111111110000000"
}
# Cursor
oldCursor=0
cursorColor=color(200,180,0)
selectColor=color(100,100,100)
cursorPos=0
cursorSize=2
select=False
selectPos = 0

def reset():
  global position
  for s in range(64):
    updateSqare(s)
  cursor()    

def calculateX(index):
  return int(((index/8)-floor(index/8))*8*squareSize+gridX)

def calculateY(index):
  return floor(index/8)*squareSize+gridY

def positionPos(square):
  i=0
  x=0
  while x<=square:
    if position[i]!="/" and position[i]!="1" and position[i]!="2" and position[i]!="3" and position[i]!="4" and position[i]!="5" and position[i]!="6" and position[i]!="7" and position[i]!="8":
      x+=1
    elif position[i]=="1" or position[i]=="2" or position[i]=="3" or position[i]=="4" or position[i]=="5" or position[i]=="6" or position[i]=="7" or position[i]=="8":
      x+=int(position[i])
    else:
      x=x
    i+=1
  i-=1
  return i

def getPiece(square):
  i=positionPos(square)
  if position[i]!="/" and position[i]!="1" and position[i]!="2" and position[i]!="3" and position[i]!="4" and position[i]!="5" and position[i]!="6" and position[i]!="7" and position[i]!="8":  
    return position[i]
  else:
    return position[i]

def updateSqare(square):
  global position
  global gColor1
  global gColor2
  if square%2==1 and floor(calculateY(square)/squareSize)%2==1:
    c=gColor1
  elif square%2==0 and floor(calculateY(square)/squareSize)%2==0:
    c=gColor1
  else:
    c=gColor2
  fill_rect(calculateX(square),calculateY(square),squareSize,squareSize,c)
  drawPiece(square, getPiece(square))
  

def drawPiece(square, p):
  if p=="r" or p=="R":
    if p!="r":
      pieces.rook("white",square)
    else:
      pieces.rook(color(80,80,80),square)
  elif p=="p" or p=="P":
    if p!="p":
      pieces.pawn("white",square)
    else:
      pieces.pawn(color(80,80,80),square)
  elif p=="n" or p=="N":
    if p!="n":
      pieces.knight("white",square)
    else:
      pieces.knight(color(80,80,80),square)
  elif p=="b" or p=="B":
    if p!="b":
      pieces.bishop("white",square)
    else:
      pieces.bishop(color(80,80,80),square)
  elif p=="q" or p=="Q":
    if p!="q":
      pieces.queen("white",square)
    else:
      pieces.queen(color(80,80,80),square)
  elif p=="k" or p=="K":
    if p!="k":
      pieces.king("white",square)
    else:
      pieces.king(color(80,80,80),square)
  else:
    return

def cursor():
  global cursorPos
  global cursorSize
  global cursorColor
  global selectColor
  global select
  x=calculateX(cursorPos)
  y=calculateY(cursorPos)
  if select==False:
    c=cursorColor
  else:
    c=selectColor
  fill_rect(x,y,cursorSize,squareSize,c)
  fill_rect(x+squareSize-cursorSize,y,cursorSize,squareSize,c)
  fill_rect(x,y,squareSize,cursorSize,c)
  fill_rect(x,y+squareSize-cursorSize,squareSize,cursorSize,c)

def updatePos(old, new):
  tPos=""
  b=0
  for i in range(64):
    p=getPiece(i)
    if i!=old and i!=new:
      if p=="1" or p=="2" or p=="3" or p=="4" or p=="5" or p=="6" or p=="7" or p=="8":
        b+=1
      elif p!="1" and p!="2" and p!="3" and p!="4" and p!="5" and p!="6" and p!="7" and p!="8":
        if b>0:  
          tPos=tPos+str(b)+p
          b=0
        elif b==0:
          tPos=tPos+p
    else:
      if i==new and b>0:
        tPos=tPos+str(b)+getPiece(old)
        b=0
      elif i==new:
        tPos=tPos+getPiece(old)
      else:
        b+=1
    if (i+1)%8==0 and i<63:
      if b>0:
        tPos=tPos+str(b)+"/"
      else:
        tPos=tPos+"/"
      b=0
  global position
  position=tPos

def movePiece(old, new):
  global player
  p=getPiece(old)
  if validate.main(old, new)==True and old!=new:
    updatePos(old, new)
    if player=="w":
      player="b"
    else:
      player="w"
  updateSqare(selectPos)
  updateSqare(cursorPos)

class validate:
  def main(old, new):
    vItems=[validate.correctColor(old), validate.pieceMoved(getPiece(old), old, new)]
    for item in vItems:
      if item==False:
        return False
    return True
  
  def pieceMoved(p, old, new):
    if p=="p" or p=="P":
      if p=="p":
        if getPiece(new)!="p" or getPiece(new)!="q" or getPiece(new)!="k" or getPiece(new)!="n" or getPiece(new)!="r" or getPiece(new)!="b":
          if (old-new)==-8:
            return True
          elif old>=48 and old<=55 and (old-new)==-16:
            return True
          elif getPiece(new)=="P" or getPiece(new)=="N" or getPiece(new)=="K" or getPiece(new)=="Q" or getPiece(new)=="R" or getPiece(new)=="B":
            if (old-new)==-7 or (old-new)==-9:
              return True
          elif old>=24 or old<=31:
            if (old-new)==-7 or (old-new)==-9:
              return True
          else:
            return False
          
      else:
        if old<=55 and old>=48:
          if (old-new)==(16):
            return True
        if (old-new)==(8):
          return True
        elif getPiece(new)=="p" or getPiece(new)=="k" or getPiece(new)=="r" or getPiece(new)=="b" or getPiece(new)=="n" or getPiece(new)=="q":
          if (old-new)==(7) or (old-new)==(9):
            return True
          else:
            return False
        else:
          return False
    elif p=="n" or p=="N":
      knightMoves=[ (10), (-10), (6), (-6), (17), (-17), (15), (-15) ]
      for move in knightMoves:
        if (old-new)==move:
          return True
      return False
    elif p=="b" or p=="B":
      if (old-new)%17==0 or (old-new)%15==0:
        return True
      else:
        return False
    
  def correctColor(old):
    if player=="w":
      if getPiece(old)=="r" or getPiece(old)=="b" or getPiece(old)=="n" or getPiece(old)=="k" or getPiece(old)=="q" or getPiece(old)=="p":
        return False
      else:
        return True
    elif player=="b":
      if getPiece(old)=="r" or getPiece(old)=="b" or getPiece(old)=="n" or getPiece(old)=="k" or getPiece(old)=="q" or getPiece(old)=="p":
        return True
      else:
        return False

class pieces:
  def pawn(color,square):    
    y=calculateY(square)
    x=calculateX(square)-1+squareSize
    sprite=str(sprites["pawn"])

    for i in range(len(sprite)):
      if sprite[i]=="1":
        set_pixel(x,y,color)
      if i%22==0:
        x-=21
        y+=1
      else:
        x+=1
  def rook(color,square):
    y=calculateY(square)
    x=calculateX(square)-1+squareSize
    sprite=str(sprites["rook"])

    for i in range(len(sprite)):
      if sprite[i]=="1":
        set_pixel(x,y,color)
      if i%22==0:
        x-=21
        y+=1
      else:
        x+=1
  def knight(color,square):
    y=calculateY(square)
    x=calculateX(square)-1+squareSize
    sprite=str(sprites["knight"])

    for i in range(len(sprite)):
      if sprite[i]=="1":
        set_pixel(x,y,color)
      if i%22==0:
        x-=21
        y+=1
      else:
        x+=1
  def bishop(color,square):
    y=calculateY(square)
    x=calculateX(square)-1+squareSize
    sprite=str(sprites["bishop"])

    for i in range(len(sprite)):
      if sprite[i]=="1":
        set_pixel(x,y,color)
      if i%22==0:
        x-=21
        y+=1
      else:
        x+=1
  def queen(color,square):
    y=calculateY(square)
    x=calculateX(square)-1+squareSize
    sprite=str(sprites["queen"])

    for i in range(len(sprite)):
      if sprite[i]=="1":
        set_pixel(x,y,color)
      if i%22==0:
        x-=21
        y+=1
      else:
        x+=1
  def king(color,square):
    y=calculateY(square)
    x=calculateX(square)-1+squareSize
    sprite=str(sprites["king"])

    for i in range(len(sprite)):
      if sprite[i]=="1":
        set_pixel(x,y,color)
      if i%22==0:
        x-=21
        y+=1
      else:
        x+=1


def update():
  global oldCursor
  global cursorPos
  global selectPos
  if select==True and oldCursor == selectPos:
    selectPos=oldCursor
  else:
    updateSqare(oldCursor)
  oldCursor=cursorPos  
  cursor()

reset()

while playing:
  if keydown(KEY_RIGHT):
    if (cursorPos+1)%8==0:
      cursorPos -= 7
    else:
      cursorPos += 1
    if cursorPos<64:
      update()
  elif keydown(KEY_DOWN):
    if (cursorPos+8)>63:
      cursorPos -= 56
    else:
      cursorPos += 8
    update()
  elif keydown(KEY_UP):
    if (cursorPos-8)<0:
      cursorPos += 56
    else:
      cursorPos -= 8
    update()
  elif keydown(KEY_LEFT):
    if (cursorPos)%8==0:
      cursorPos += 7
    else:
      cursorPos -= 1
    update()
  elif keydown(KEY_OK):
    if select==False:
      select=True
      selectPos=cursorPos
    else:
      select=False
      movePiece(selectPos, cursorPos)
    cursor()

  sleep(.15)
