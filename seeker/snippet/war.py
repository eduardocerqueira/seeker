#date: 2021-11-11T17:16:03Z
#url: https://api.github.com/gists/b58aa9edd22a3c0cf38f3258bc2d860d
#owner: https://api.github.com/users/kpessa

# Here's my python solution to the 'War' - queue challenge on CodeInGame
# https://www.codingame.com/ide/puzzle/winamax-battle
import sys
from collections import deque

p1 = deque()
p2 = deque()
p1bounty = deque()
p2bounty = deque()

n = int(input())  # the number of cards for player 1
for i in range(n):
    cardp_1 = input()  # the n cards of player 1
    p1.append(cardp_1)
m = int(input())  # the number of cards for player 2
for i in range(m):
    cardp_2 = input()  # the m cards of player 2
    p2.append(cardp_2)

# To debug: print("Debug messages...", file=sys.stderr, flush=True)
# print("PAT")

# -----------------------------------------------
#             UTIILITY FUNCTIONS
# -----------------------------------------------
def getCardValue(card):
    value = card[:-1]
    if value in "JQKA":
        if value == 'J': return 11
        if value == 'Q': return 12
        if value == 'K': return 13
        if value == 'A': return 14
    else:
        return int(value)
gcv=getCardValue

def war(p1,p2):
    for _ in range(3): 
        if len(p1) > 1: p1bounty.append(p1.popleft())
        if len(p2) > 1: p2bounty.append(p2.popleft())

def win(player):
    if player == 'player1':
        while(p1bounty): p1.append(p1bounty.popleft())
        while(p2bounty): p1.append(p2bounty.popleft())
    else:
        while(p1bounty): p2.append(p1bounty.popleft())
        while(p2bounty): p2.append(p2bounty.popleft())

def printStatus(p1card,p2card,round):
    print(f"Round {round}: {p1card} vs {p2card}", file=sys.stderr, flush=True)
    print(f"-- {gcv(p1card)} vs {gcv(p2card)}", file=sys.stderr, flush=True)
    print(f"-- p1:{len(p1)}, p2:{len(p2)}", file=sys.stderr, flush=True)

# -----------------------------------------------
#             MAIN GAME LOOP
# -----------------------------------------------

def game():
    round = 0
    while(p1 and p2):
        round = round + 1
        p1card = p1.popleft(); p1bounty.append(p1card); p1value = getCardValue(p1card)
        p2card = p2.popleft(); p2bounty.append(p2card); p2value = getCardValue(p2card)

        # printStatus(p1card,p2card,round)

        if p1value > p2value: 
            win('player1')
        elif p1value < p2value: 
            win('player2')
        else:
            # print(f"****WAR*****", file=sys.stderr, flush=True)
            if(len(p1) < 3 or len(p2) < 3): return "PAT"
            if(p1 and p2): 
                war(p1,p2)
                round = round-1
            

        print(f"", file=sys.stderr, flush=True)
        
        # print(getCardValue(p1card))
        # print(getCardValue(p2card))
    
    if p1: return f"1 {round}"
    else: return f"2 {round}"

print(game())