#date: 2023-07-27T17:06:05Z
#url: https://api.github.com/gists/86c24d88398e346a73e371b45fd9b21a
#owner: https://api.github.com/users/SightSpirit

'''
Description:
I got bored and decided to make a class in Python. The `Crank` class represents a turnable crank and tracks how many degrees it's been turned and how many times it has been fully turned. (I may or may not have been inspired subconsciously by *Unbreakable Kimmy Schmidt*.)

Example Use Case:
I genuinely do not know, but I am genuinely excited to find out how someone uses this!

License:
This code is in the public domain. You may use it for any purpose that is considered legal in your jurisdiction, including for-profit purposes, without providing attribution or including a similar license. You may attribute to Eden Biskin, if desired.
'''

from math import floor

class Crank():
    def __init__(self):
        self.degrees = 0  # How many degrees crank has been turned clockwise *in the current rotation*
        self.turns = 0  # How many full rotations the crank has been turned clockwise
        
    # Turn crank clockwise
    def crank(self,degrees:int):
        if degrees < 0:
            raise ValueError('To crank counterclockwise, use Crank.uncrank.')
        elif degrees == 0:
            print('You grab the crank and then let go without having moved it. You\'re weird.')
            return

        self.degrees += degrees
        if self.degrees > 359:
            self.turns += floor(self.degrees / 360)
            self.degrees = self.degrees % 360

    # Turn crank counterclockwise
    def uncrank(self,degrees:int):
        if degrees < 0:
            raise ValueError('To crank clockwise, use Crank.crank.')
        elif degrees == 0:
            print('You grab the crank and then let go without having moved it. You\'re weird.')
            return

        if self.degrees - degrees > -1:
            self.degrees -= degrees
        elif self.totalDegrees - degrees < 0:
            print('Crank stopped at initial position.')
            self.turns = 0
            self.degrees = 0
        else:
            total = self.totalDegrees
            total -= degrees
            self.turns = floor(total / 360)
            self.degrees = total % 360

    # How many degrees the crank has been turned, in total
    @property
    def totalDegrees(self):
        return 360* self.turns + self.degrees

    # Returns a tuple containing how many turns and how many degrees turned in the current rotation
    @property
    def turnsDegs(self):
        return (self.turns,self.degrees)

    # Returns a floating-point number representing exactly how many times the crank has been turned clockwise
    @property
    def turnsFloat(self):
        return self.totalDegrees / 360
