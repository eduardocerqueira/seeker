#date: 2022-01-10T17:18:01Z
#url: https://api.github.com/gists/c6869fd1e16faf0ff52eeb53131caf73
#owner: https://api.github.com/users/RandomGuy3015


# Genuary 2022 - u/LordTachankaMain (Discord: Falafel#3169)
# Coded on the 05/01/2022 from 4:00-6:30PM
# This is my submission for Genuary 2022_5, "Destroy a square"

# There is somehow a memory leak in this program, and it will use ~50mb 
# of ram every time it's run, even when the program is stopped. Don't
# I have no clue. Just re-open Processing if it starts getting laggy.

actors = [] # location and vector of every point 


def setup():
    size(3000,2000)
    background(0)
    stroke(255)
    global actors
    for i in range(10000):  # initializing all the points
        actor = [0,0,0,0,False]   # 0 = x; 1 = y; 2 = delta x (velocity); 3 = delta y; 4 = wasTouchedByCursor.
        actor[0] = int(random(0,3000))
        actor[1] = int(random(0,2000))
        actors.append(actor)
        

def draw():
    #delay(200)
    background(0)
    for actor in actors:
        square(actor[0],actor[1],5)  # draws the points 
        move(actor)                  # moves the points by a small amount
        dis = tria(actor)            # calculates the distance to cursor
        
        if dis < 120000:            # if point in radius around cursor...
            move(actor)
            disNew = tria(actor)
            if disNew < dis:          # if, after moving again, the point is closer to the cursor...
                factor = int(60/(float(dis*0.0002)))  # the closer the point is to the cursor,
                if actor[2] > 0:                      #  the more biased it is to turn around
                    actor[2] -= factor
                else: 
                    actor[2] += factor
                if actor[3] > 0:
                    actor[3] -= factor
                else: 
                    actor[3] += factor
            actor[4] = True
                
        if actor[4] == False:     # if actor was affected by cursor, make it unaffected by square borders
            if actor[0] > 2000:   # all of the 'out of square' checks
                actor[0] = 1000
                actor[2] *= 0.5    # slows the points down so they don't keep getting faster
                actor[3] *= 0.5
            
            if actor[0] < 1000:
                actor[0] = 2000
                actor[2] *= 0.5
                actor[3] *= 0.5
                
            if actor[1] > 1500:
                actor[1] = 500
                actor[2] *= 0.5
                actor[3] *= 0.5
            
            if actor[1] < 500:
                actor[1] = 1500
                actor[2] *= 0.5
                actor[3] *= 0.5
            
    

            
def tria(actor):    # uses pythogoras for the distance
    global actors
    dist = (actor[0] - mouseX)**2 + (actor[1] - mouseY)**2
    return dist

def move(actor):    # adds or subtracts to the velocity in a random direction
    x = int(random(-2,2)) + actor[2]
    y = int(random(-2,2)) + actor[3]
    actor[0] += x
    actor[1] += y
    actor[2] = x
    actor[3] = y
    
    