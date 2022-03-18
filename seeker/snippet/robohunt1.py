#date: 2022-03-18T17:12:19Z
#url: https://api.github.com/gists/f6390bf3ad113d09ba7ee9b0f187b259
#owner: https://api.github.com/users/horstjens

# robots hunting the player
# try to escape, robots are destroyed if they 
# collide with each other or with an wall

import random


class Monster:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.char = ANGRY

    def hunt(self, target):
        dx, dy = 0, 0
        if self.x < target.x:
            dx = 1
        elif self.x > target.x:
            dx = -1
        if self.y < target.y:
            dy = 1
        elif self.y > target.y:
            dy = -1
        return dx, dy



#ANGRY = "\U0001F620" # 😠
#NICE  = "\U0001F603" # 😃
#BOOM  = "\U0001F4A5" # 💥

ANGRY = "Ö"
NICE = "@"
BOOM = "x"
KAPUTT = "X"

playfield = """
################################################
#..............................................#
#..............................................#
#..............................................#
#..............................................#
#..............................................#
#..............................................#
#..............................................#
#..............................................#
#..............................................#
#..............................................#
#..............................................#
################################################
"""
area = [list(line) for line in playfield.splitlines() if len(line.strip())>0]
width = len(area[0])
height = len(area)
middle_y = height // 2
middle_x = width //2 


player = Monster(middle_x,middle_y)
player.char = NICE

everyone = [player] #,  Monster(3,1), Monster(7,1), Monster(1,3), Monster(8,11), Monster(46,5)]


def place_random_walls(howmany = 16):
    replaced = 0
    # area is a list of lists
    for i in range(howmany):
        for j in range(1000):
            # try 1000 times or give up
            y = random.randint(1, height - 2)
            x = random.randint(1, width - 2)
            if (x==player.x) and (y==player.y):
                continue
            if area[y][x] == ".":
                area[y][x] = "#"
                replaced += 1
                break
        else:
            print(f"random wall {j}: unable to found free position in area")


def place_random_monsters(howmany = 8, max_distance_from_border = 1):
    # area is a list of lists
    # only place monster on spots with a dot "."
    possible_positions = set()
    # north wall
    for y in range(1,  max_distance_from_border+1 ):
        for x in range(1, width-2 - max_distance_from_border):
            possible_positions.add((x,y))
    # south wall
    for y in range(height - 1  - max_distance_from_border, height-1):
        for x in range(1, width-3 - max_distance_from_border):
            possible_positions.add((x,y))
    # west wall
    for y in range(1, height-2):
        for x in range(1, max_distance_from_border+1):
            possible_positions.add((x,y))
    # east wall
    for y in range(1, height-2):
        for x in range(width-max_distance_from_border - 1, width-1):
            possible_positions.add((x,y))

    positions = random.sample(list(possible_positions), howmany)
    print(width, height, positions)
    for pos in positions:
        x, y = pos
        everyone.append(Monster(x,y))



def wallcheck(x,y):
    if area[y][x] == "#":
        return True
    #else:
    return False

def paint():
    for y, line in enumerate(area):
        for x, char in enumerate(line):
            # monster/player at this position?
            for thing in everyone:
                if (thing.x == x) and (thing.y == y):
                    print(thing.char, end="")
                    break
            else:
                print(char, end="")
        print() # new line




place_random_walls()  

place_random_monsters(8,1)

message = ""
while player.char == NICE:
    paint()
    print(message)
    command = input("move with wasd, or type 'quit': >>>").lower().strip()
    message = ""
    dx, dy = 0,0
    if command == "quit":
        break
    if command in ("w", "8"):
        dy = -1
    if command in ("s", "2"):
        dy = 1
    if command in ("a", "4"):
        dx = -1
    if command in ("d", "6"):
        dx = 1
    if command in ("q", "7"):
        dx, dy = -1, -1
    if command in ("e", "9"):
        dx, dy = 1, -1
    if command in ("y", "1"):
        dx, dy = -1, 1
    if command in ("c", "3"):
        dx, dy = 1, 1


    # wall ? 
    if wallcheck(player.x+dx, player.y+dy):
        dx, dy = 0, 0
        message += "you run into a wall. Ouch!"
    # other monster?
    for monster in everyone: 
        if monster == player: 
            continue # everyone except the player
        if (player.x + dx == monster.x) and (player.y + dy == monster.y):
            message+="you run into a monster."
            player.char = KAPUTT
            dx, dy = 0, 0
            break
    # ------ update player
    player.x += dx
    player.y += dy
    
    
    # ---- move monsters
    for monster in everyone:
        if monster == player:
            continue # everyone except player
        if monster.char == BOOM:
            continue # next monster
        mx,my = monster.hunt(player)
        # collide with wall? 
        if wallcheck(monster.x+mx, monster.y + my):
            monster.char = BOOM
            mx, my = 0, 0
            continue # next monster
        # collide with other montser?
        for monster2 in everyone:
            if monster2 == monster :
                continue # don't test yourself
            if ((monster.x + mx) == monster2.x ) and ((monster.y + my) == monster2.y):
                monster.char = BOOM            
                monster2.char = BOOM
                if monster2 == player:
                    message += "the monster got you."
                    player.char = KAPUTT
                mx, my = 0, 0                
    
                
        monster.x += mx
        monster.y += my
    # win ? 
    survivors = [monster for monster in everyone if monster.char not in (BOOM, KAPUTT)]
    if (len(survivors) == 1) and (survivors[0] == player):
        message = "Du hast gewonnen!"
        break
    
paint()
print(message)
print("Game Over")



    

