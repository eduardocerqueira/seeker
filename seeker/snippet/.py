#date: 2021-11-23T17:13:00Z
#url: https://api.github.com/gists/7b35b9215965e57d11187f92caa29f69
#owner: https://api.github.com/users/TotalBro

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 09:32:04 2021

@author: kschroeder
"""

import random
import numpy

def defaultdeck():
    return ['+0', '+0', '+0', '+0', '+0', '+0', '+1', '+1', '+1', '+1', '+1', '-1', '-1', '-1', '-1', '-1', '+2', '-2', 'x2', 'Nil']

deck = defaultdeck()
runs = 0
results = []
stufflist = [2, deck, runs]

def drawcard(roll, deck, runs, draw):            
    if deck[draw][:1] == '+':
        roll += int(deck[draw][1:])
    elif deck[draw][:1] == '-':
        roll -= int(deck[draw][1:])
    elif deck[draw][:1] == 'x':
        roll = roll*2
        deck = defaultdeck()
    elif deck[draw] == 'Nil':
        roll = 0
        deck = defaultdeck()
    elif deck[draw][:1] == 'U':
        deck[draw] = 'C'
        drawcard(roll, deck, runs, draw)
    elif deck[draw] == 'C':
        runs -= 1
    else:
        print('I havent programmed that route yet')
    
    if deck[draw] not in ['C', 'U']:
        results.append(roll)
    deck[draw] = 'C'
    
    return [roll, deck, runs]
    
while stufflist[2] < 1000000:
    draw = random.randrange(len(deck))
    stufflist[0] = 2
    stufflist = drawcard(stufflist[0], stufflist[1], stufflist[2], draw)
    
    results.append(stufflist[0])
    stufflist[2] += 1    
    
print(numpy.mean(results))