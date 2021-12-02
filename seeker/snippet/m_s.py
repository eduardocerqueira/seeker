#date: 2021-12-02T16:56:14Z
#url: https://api.github.com/gists/526caf5913a1e5abca568f51707d87ef
#owner: https://api.github.com/users/OJddJO

from math import *
from kandinsky import *
from ion import *
from random import *
from time import *

# implementer save.py
# mettre la ligne suivante dans save.py

money=0;pickaxe="wood_p";sword="wood_s"

fill_rect(0,0,320,222,'black')
draw_string("MINING SIMULATOR",80,40)
draw_string("[OK]:PLAY",115,160)
draw_string("2x[RETURN]:EXIT",83,183)

run = 1
ig_money = money
ig_pick = pickaxe
ig_sword = sword
loaded = "start"

def saving():
    print("Replace the text in the file")
    print("save.py with this text to save")
    print("your progression :")
    print("(Copy the text with [SHIFT]+[VAR])")
    print("money="+str(ig_money)+";pickaxe="+str(ig_pick)+";sword="+str(ig_sword))


while run==1:
    sleep(0.1)
    if keydown(KEY_OK):
        fill_rect(0,0,360,240,'black')
        run = 2

while run == 2:
    if loaded=="start":
        draw_string("MONEY : " + str(ig_money),10,5)
        draw_string("[EXE]:Save and Exit",60,200)
        fill_rect(170,30,145,150,'white')
        draw_string("INFO :",170,30)
# 14 caractere par option
        draw_string("[1]:Mine      ",10,30)
        draw_string("[2]:Fight     ",10,48)
        sleep(0.1)
        loaded = "menu"

    if loaded=="menu":
# minage
        if keydown(KEY_ONE):
            if ig_pick=="wood_p":
                loot = 1
            if ig_pick=="stone_p":
                loot = 2
            if ig_pick=="iron_p":
                loot = 3
            if ig_pick=="diamond_p":
                loot = 4
            draw_string("You mined",170,48)
# combat

# loot msg
        item_mining = ["stone","coal","iron","gold","diamond"]
        item_fight = ["lapis","leather"]


    if keydown(KEY_EXE):
        saving()
        fill_rect(0,0,360,240,'black')
        draw_string("CLICK ON [RETURN]",80,100)
        run = 0
