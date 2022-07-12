#date: 2022-07-12T16:55:27Z
#url: https://api.github.com/gists/95b3e3e92ac4b88a80d17e4805a5b679
#owner: https://api.github.com/users/enderbob364

# -*- coding: utf-8 -*- #

class Game(object):

    def __init__(self):
        self.rolls = []

    def roll(self, pins):
        self.rolls.append(pins)

    def total_score(self):
        score = 0
        roll_index = 0
        for frame in range(10):
            if self._is_strike(roll_index):
                score += 10 + self.rolls[roll_index+1] + self.rolls[roll_index+2]
                roll_index += 1
            elif self._is_spare(roll_index):
                score += 10 + self.rolls[roll_index+2]
                roll_index += 2
            else:
                score += self.rolls[roll_index] + self.rolls[roll_index+1]
                roll_index += 2
        return score

    def _is_spare(self, roll_index):
        return self.rolls[roll_index] + self.rolls[roll_index+1] == 10

    def _is_strike(self, roll_index):
        return self.rolls[roll_index] == 10
