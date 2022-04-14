#date: 2022-04-14T16:49:12Z
#url: https://api.github.com/gists/36ef54a620bc6409cab137315efb1ca8
#owner: https://api.github.com/users/samneggs

#Lunar Lander 1 bit plane

import gc9a01
from machine import Pin, SPI, PWM, WDT
import framebuf
from time import sleep_ms, sleep_us, ticks_diff, ticks_us, sleep
from micropython import const
import array
from usys import exit
from math import sin,cos,pi,radians,tan, degrees, atan2
from uctypes import addressof
from random import randint
import gc
import _thread


MAXSCREEN_X = const(240)
MAXSCREEN_Y = const(240)
MAXCHUNK_Y=const(80)
SCALE = const(10)
LAND_MAX = const(200)

joyRight = Pin(17,Pin.IN)
joyDown  = Pin(18,Pin.IN)
joySel   = Pin(19,Pin.IN)
joyLeft  = Pin(20,Pin.IN)
joyUp    = Pin(21,Pin.IN)

class Ship():
    def __init__(self):
        self.x = 120
        self.y = 20
        self.dx = 0
        self.dy = 0
        self.a = 225  # angle
        self.h = 20  # height
        self.w = 120
        self.s = .2   # size
        self.s_v = int(.2 * (1<<SCALE))
        self.x_v = 120 * (1<<SCALE)
        self.y_v = 20 * (1<<SCALE)
        self.dx_v = 0
        self.dy_v = 0
        self.h_v = 20 * (1<<SCALE)
        self.w_v = 120 * (1<<SCALE)
        self.landpos = 0
        self.crash = 0
        self.landed = 0
    def crashed(self):
        self.crash = 1
    def land(self):
        self.dx_v = 0
        self.dy_v = 0
        self.landed = 1
        
ship=Ship()

print('pre alloc',gc.mem_alloc())
print('initial free',gc.mem_free())


ship_cart=[-6,0,0, -14,8,0, -14,20,0, -6,29,0, 6,29,0, 14,20,0, 14,20,0, 14,8,0, 6,0,0, -6,0,1,
      -17,0,0, -17,-16,0 ,17,-16,0 ,17,0,0, -17,0,0, -32,-24,1,
       17,0,0,  32,-24,1,
       17,-14,1, -28,-18,1, 17,-14,1, 28,-18,1,
       36,-24,0,  28,-24,1,
      -28,-24,0, -36,-24,1,
       -3,-16,0,  -7,-21,0, 7,-21,0, 3,-16,1,
       -6,-24,0,  -20,-50,0,  0, -120,0 , 20, -50,0, 6,-24,0]
        
def init_ship():    
    data=array.array('i',(6,180,0 ,16,150,0 ,24,124,0 ,29,101,0 ,29,78,0 ,24,55,0 ,24,55,0 ,16,29,0 ,6,0,0 ,6,180,1,
      17,180,0 ,23,-136,0,23,-43,0,17,0,0,17,180,0,40,-143,1,
      17,0,0,40,-36,1 ,
      22,-39,1 ,33,-147,1 ,22,-39,1,33,-32,1,
      43,-33,0,36,-40,1 ,
      36,-139,0,43,-146,1 ,
      16,-100,0 ,22,-108,0 ,22,-71,0,16,-79,1,
      24,-104,0,     53,-111,0,   120,-90,0,   53,-68,0, 24,-75,0))
    return data


land_rad=array.array('i',0 for _ in range(LAND_MAX))
land_deg=array.array('i',0 for _ in range(LAND_MAX))
isin=array.array('i',range(0,360))
icos=array.array('i',range(0,360))

char_map=array.array('b',(
     0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00,   # U+0030 (0)
     0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00,   # U+0031 (1)
     0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00,   # U+0032 (2)
     0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00,   # U+0033 (3)
     0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00,   # U+0034 (4)
     0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00,   # U+0035 (5)
     0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00,   # U+0036 (6)
     0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00,   # U+0037 (7)
     0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00,   # U+0038 (8)
     0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00))  # U+0039 (9)


#one color
@micropython.asm_thumb
def chunk_asm1(r0): # 0=from, 4=to, 8=offset, 12=#bytes, 16=color
    ldr(r1,[r0,0])   # r1=screen addr
    ldr(r2,[r0,8])   # offset
    add(r1,r1,r2)    # r1=screen addr+offset
    ldr(r2,[r0,4])   # r2=chunk addr
    ldr(r3,[r0,12])  # r3=#bytes
    ldr(r4,[r0,16])  # r4=color   
    label(LOOP)
    ldrb(r5,[r1,0])  # r5=load screen byte
    mov(r6,8)        # r6=bit count
    label(BYTE)    
    asr(r5,r5,1)     # next bit
    bcs(ONE)
    mov(r7,0)        # color = 0 
    b(NEXT)
    label(ONE)
    mov(r7,r4)       # color = 1 
    label(NEXT)
    strh(r7,[r2,0])  # write chunk byte
    add(r2,r2,2)     # next chunk byte
    sub(r6,r6,1)     # next bit
    bne(BYTE)
    add(r1,r1,1)     # next screen byte
    sub(r3,r3,1)     # number bytes -1
    bne(LOOP)
    label(EXIT)
    

@micropython.asm_thumb
def cls_asm1(r0): # screen
    mov(r1,225)
    lsl(r1,r1,5) # 7200, screen size
    mov(r2,0)
    label(LOOP)
    strb(r2,[r0,0])
    add(r0,1)
    sub(r1,1)
    bne(LOOP)
    label(EXIT)

def show1():
    chunk_ctl[2]=0
    chunk_asm1(chunk_ctl)
    screen_fps(fps,306)
    screen_fps(ship.a,310)
    screen_fps(ship.dx_v//100,580)
    screen_fps(ship.dy_v//100,880)
    tft.blit_buffer(chunk, 0, 0, MAXSCREEN_X, MAXCHUNK_Y)
    chunk_ctl[2]=2400
    chunk_asm1(chunk_ctl) 
    tft.blit_buffer(chunk, 0, 80, MAXSCREEN_X, MAXCHUNK_Y)
    chunk_ctl[2]=2400*2
    chunk_asm1(chunk_ctl)
    tft.blit_buffer(chunk, 0, 160, MAXSCREEN_X, MAXCHUNK_Y)

@micropython.viper
def screen_fps(n:int,p:int):
    dest=ptr16(chunk)
    c_map=ptr8(char_map)
    huns = n//100
    tens = n%100//10 #int((0xcc_cccd*n)>>27) # n//10
    ones = n%10 #n-(tens*10)             # n%10
    row  = 8 
    offset=p*8+50
    while(row):
        row-=1
        col=8
        while(col):
            col-=1
            r=(row*MAXSCREEN_X+col)+offset
            if huns>0 and c_map[(huns<<3)+row] & 1<<col:
                dest[r] =  0xffff 
            if c_map[(tens<<3)+row] & 1<<col:
                dest[r+8] =  0xffff 
            if c_map[(ones<<3)+row] & 1<<col:
                dest[r+16] = 0xffff

def init_isin():            # integer sin lookup table    
    for i in range(0,360):
        isin[i]=int(sin(radians(i))*(1<<SCALE))
            

def init_icos():            # integer cos lookup table 
    for i in range(0,360):
        icos[i]=int(cos(radians(i))*(1<<SCALE))

def init_game():
    global ship_data
    ship.__init__()
    init_land()
    ship_data=init_ship()    

def init_land():
    land_rad[0]=0
    land_deg[0]=200
    t=3
    y=0
    x=-400   #-250
    i=0
    pad=randint(0,50) #LAND_MAX)
    while i<LAND_MAX:
        if i==pad and ship.landpos==0:
            x+=100
            ship.landpos = i-1
            #print('landing',ship.landpos)
            y-=10
        else:
            x+=20  #5
            y=y+randint(-20,20)
        land_rad[i]=int((x ** 2 + y ** 2) ** .5)
        land_deg[i]=int(degrees(atan2(y,x)))
        #print(land_rad[i],land_deg[i])
        i+=1


@micropython.viper
def draw_land_v():
    vsin=ptr32(isin)
    vcos=ptr32(icos)
    vradius=ptr32(land_rad)
    vdeg=ptr32(land_deg)
    size=int(ship.s_v)
    height=(190<<SCALE)+int(ship.y_v)-int(ship.h_v)
    scroll=((120<<SCALE)+int(ship.x_v)-int(ship.w_v))#+(size-204)*200
    #scroll+=(size-204)*1
    #print(122880-scroll,size-204)
    ang=int(ship.a)
    deg=230
    j=0
    x1coll=int(ship.x_v)-size*20
    y1coll=int(ship.y_v)-size*40
    x2coll=int(ship.x_v)+size*25
    y2coll=int(ship.y_v)+size*35
    #screen.rect(x1coll,y1coll,x2coll-x1coll,y2coll-y1coll,0xffff)
    for i in range(1,LAND_MAX-1):
      
        r=vradius[i]*size
        d=vdeg[i]+deg
        if d>359:
            d-=360
        if d<0:
            d+=360
        x1=((r*vsin[d])>>SCALE)+((r*vcos[d])>>SCALE)+scroll
        y1=((r*vsin[d])>>SCALE)-((r*vcos[d])>>SCALE)+height
        r=vradius[i+1]*size
        d=vdeg[i+1]+deg
        if d>359:
            d-=360
        if d<0:
            d+=360
        x2=((r*vsin[d])>>SCALE)+((r*vcos[d])>>SCALE)+scroll
        y2=((r*vsin[d])>>SCALE)-((r*vcos[d])>>SCALE)+height
        if x1<0 or x2<0 or x1>240<<SCALE or x2>240<<SCALE or y1<0 or y2<0 or y1>240<<SCALE or y2>240<<SCALE:
            continue
        j+=1
        if i==int(ship.landpos):
            color=0xf8
        else:
            color=0xffff
        screen.line(x1>>SCALE,y1>>SCALE,x2>>SCALE,y2>>SCALE,color)       
        if int(ship.crash)==0 and int(ship.landed)==0 and x1coll<x1 and x2coll>x2 and y1coll<y1 and y2coll>y2:
            if i==int(ship.landpos) and ang>220 and ang<230:
                print('landed',ship.a)
                ship.land()
            else:
                print('collision',ship.a)
                ship.crashed()
                
                   
def convert():
    rang=range(0,len(ship_cart)-0,3)
    for i in rang:
        x=ship_cart[i]
        y=ship_cart[i+1]
        r = int((x ** 2 + y ** 2) ** .5)
        theta = int(degrees(atan2(y,x)))
        print(x,y,r,',',theta,',0,')


@micropython.viper
def draw_ship_v():
    vscreen=ptr16(screen)
    vsin=ptr32(isin)
    vcos=ptr32(icos)
    vship=ptr32(ship_data)
    deg=int(ship.a)
    x=int(ship.x_v)
    y=int(ship.y_v)
    size=int(ship.s_v)
    rng=int(len(ship_data))
    for i in range(0,rng-3,3):
        r=vship[i]*size
        d=vship[i+1]+deg
        if d>359:
            d-=360
        if d<0:
            d+=360
        x1=((r*vsin[d])>>SCALE)+((r*vcos[d])>>SCALE)+x
        y1=((r*vsin[d])>>SCALE)-((r*vcos[d])>>SCALE)+y
        r=vship[i+3]*size
        d=vship[i+4]+deg
        if d>359:
            d-=360
        if d<0:
            d+=360
        x2=((r*vsin[d])>>SCALE)+((r*vcos[d])>>SCALE)+x
        y2=((r*vsin[d])>>SCALE)-((r*vcos[d])>>SCALE)+y
        if vship[i+2]==0:
            screen.line(x1>>SCALE,y1>>SCALE,x2>>SCALE,y2>>SCALE,0xffff)
        if int(ship.crash)>0:
            vship[i+0]+=int(randint(-2,2)) # r1
            vship[i+1]+=int(randint(-5,5)) # d1
            vship[i+3]+=int(randint(-2,2)) # r2
            vship[i+4]+=int(randint(-5,5)) # d2    

def draw():
    x=120
    y=180
    r=range(0,len(ship_cart)-3,3)
    for i in r:
        if ship_cart[i+2]==0:
            screen.line(ship_cart[i]+x,ship_cart[i+1]+y,ship_cart[i+3]+x,ship_cart[i+4]+y,0xffff)
    
def buttons2():
    if not joySel.value():
        init_game()
    if not joyRight.value():
        ship.a+=3
    if not joyLeft.value():
        ship.a-=3
    if ship.a>360:
        ship.a-=360
    if ship.a<0:
        ship.a+=360
    if not joyUp.value() and ship.crash==0:
        d=ship.a+90
        if d>360:
            d-=360
        thrust=20
        x=(isin[d]//thrust)+(icos[d]//thrust)
        y=(isin[d]//thrust)-(icos[d]//thrust)
        ship.dx_v+=x
        ship.dy_v+=y
        ship_data[101]=0
        ship_data[98]=0
        ship_data[95]=0
        ship_data[92]=0
    elif ship_data[101]==0: # engine off
        ship_data[101]=1
        ship_data[98]=1
        ship_data[95]=1
        ship_data[92]=1
    if not joyDown.value():    
        pass
    if ship.s_v<200:
        ship.s_v=200


def move_ship():
    ship.w+=ship.dx
    ship.h+=ship.dy
    if ship.h<180:
        ship.y+=ship.dy
    if ship.h>100 and ship.h<140:       # ship height to
        ship.s+=ship.dy/200             # increase scale
    if ship.w>50 and ship.w<200:
        ship.x+=ship.dx
    ship.dy+=.01
    if ship.crash>0:
        ship.crash+=1
        ship_data[ship.crash*3]=1
        if ship.crash>20:
            init_game()

def move_ship2():
    ship.w_v+=ship.dx_v
    ship.h_v+=ship.dy_v
    if ship.h_v<180<<SCALE:
        ship.y_v+=ship.dy_v
    if 1 and ship.h_v>100<<SCALE and ship.h_v<140<<SCALE:       # ship height to
        ship.s_v+=(ship.dy_v)//200                  # increase scale
    if ship.w_v>50<<SCALE and ship.w_v<200<<SCALE:
        ship.x_v+=ship.dx_v
    if not ship.landed:
        ship.dy_v+= ((1<<SCALE)//100) # gravity
    if ship.crash>0:
        ship.crash+=1
        ship_data[ship.crash*3]=1
        if ship.crash>20:
            init_game()
            
            
def print_vars():
    print(ship.x,ship.y,ship.dx,ship.dy,ship.a)

if __name__=='__main__':
    spi = SPI(1, baudrate=63_000_000, sck=Pin(10), mosi=Pin(11))
    tft = gc9a01.GC9A01(
        spi,
        MAXSCREEN_X,
        240,
        reset=Pin(12, Pin.OUT),
        cs=Pin(9, Pin.OUT),
        dc=Pin(8, Pin.OUT),
        backlight=Pin(13, Pin.OUT),
        rotation=0)
    tft.init()
    tft.rotation(0)
    tft.fill(gc9a01.BLACK) 
    chunk_buffer=bytearray(240 * MAXCHUNK_Y * 2)
    screen_buffer=bytearray(240*240//8)
    chunk=framebuf.FrameBuffer(chunk_buffer, MAXSCREEN_X , MAXCHUNK_Y, framebuf.RGB565)
    screen=framebuf.FrameBuffer(screen_buffer, MAXSCREEN_X , MAXSCREEN_Y, framebuf.MONO_HMSB) 
    print('after screen free',gc.mem_free())   
    chunk_ctl=array.array('i',(addressof(screen),addressof(chunk),2400,2400,0xffff))
    init_isin()
    init_icos()
    init_game()
    gticks=ticks_us()
    gc.collect()
    print('main loop free',gc.mem_free())
    while(1):
        buttons2()
        draw_land_v()
        draw_ship_v()
        move_ship2()       
        fps=1_000_000//ticks_diff(ticks_us(), gticks)
        gticks=ticks_us()
        show1()
        cls_asm1(screen)
        #print(gc.mem_free())
