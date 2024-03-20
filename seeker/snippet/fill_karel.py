#date: 2024-03-20T17:02:34Z
#url: https://api.github.com/gists/03e215a9c7a49529fdb47976ad5c1761
#owner: https://api.github.com/users/hamees-sayed

from karel.stanfordkarel import *

"""
Karel should fill the whole world with beepers. Solution for Stanford Code in Place 2024 Section Leader Exercise
"""
def face_east():
    """
    Karel will face east independent of what the current direction is.
    """
    if facing_north():
        turn_right()
    if facing_west():
        turn_around()
    if facing_south():
        turn_left()

def turn_around():
    """
    Turns Karel 180 degrees.
    """
    turn_left()
    turn_left()

def turn_right():
    """
    Opposite of turn_left()
    """
    for _ in range(3):
        turn_left()

def put_beeper_line():
    """
    Places beepers in a row until a wall is reached.
    """
    while front_is_clear():
        move()
        face_east()
        if no_beepers_present():
            put_beeper()

def reset_position():
    """
    Moves Karel back to the beginning of the next row.
    """
    turn_around()
    while front_is_clear():
      move()
    turn_right()
  
def settle_end():
    face_east()
    while front_is_clear():
        move()
  

def main():
    """
    Fills the world with beepers row by row.
    """
    put_beeper()
    put_beeper_line()
    reset_position()
    while front_is_clear():
        put_beeper_line()
        reset_position()
    if front_is_blocked():
        settle_end()
  
    
# There is no need to edit code beyond this point
if __name__ == '__main__':
    main()