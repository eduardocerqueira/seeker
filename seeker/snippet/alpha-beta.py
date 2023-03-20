#date: 2023-03-20T17:04:33Z
#url: https://api.github.com/gists/640b3c7bccf6d8112381f3303fa2f169
#owner: https://api.github.com/users/Aperence

import numpy as np

class Node:
    def __init__(self, childrens, payoff=0, name=""):
        self.terminal = len(childrens) == 0
        self.payoff = payoff
        self.childrens = childrens
        if (self.terminal):
            self.name = str(payoff)
        else:
            self.name = name

    def __str__(self):
        return self.name

class Game:
    def __init__(self):
        pass

    def TO_MOVE(self, state):
        return None
    
    def IS_TERMINAL(self, state : Node):
        return state.terminal

    def UTILITY(self, state : Node):
        return state.payoff

    def ACTIONS(self, state : Node):
        temp = state.childrens.copy()
        temp.reverse()
        for node in temp:
            yield node

    def RESULT(self, state : Node, a : Node):
        return a

def ALPHA_BETA_SEARCH(game, state):
    player = game.TO_MOVE(state)
    value, move = MAX_VALUE(game, state, -np.inf, np.inf)
    return move

def MAX_VALUE(game, state, alpha, beta):
    if game.IS_TERMINAL(state):
        print(F"Terminal {state}")
        return game.UTILITY(state), None
    v = -np.inf
    for a in game.ACTIONS(state):
        v2, a2 = MIN_VALUE(game, game.RESULT(state, a), alpha, beta)
        if v2 > v:
            v, move = v2, a
            alpha = max(alpha, v)
        if v >= beta :
            print(F"{state} : alpha={alpha}, beta={beta}, , value={v}")
            return v, move
        
    print(F"{state} : alpha={alpha}, beta={beta}, value={v}")
    return v, move

def MIN_VALUE(game, state, alpha, beta):
    if game.IS_TERMINAL(state):
        print(F"Terminal {state}")
        return game.UTILITY(state), None
    v = np.inf
    for a in game.ACTIONS(state):
        v2, a2 = MAX_VALUE(game, game.RESULT(state, a),alpha, beta)
        if v2 < v:
            v, move = v2, a
            beta = min(beta, v)
        if v <= alpha:
            print(F"{state} : alpha={alpha}, beta={beta}, value={v}")
            return v, move
        
    print(F"{state} : alpha={alpha}, beta={beta}, value={v}")
    return v, move

n = [Node([], 4),
     Node([], -2),
     Node([], 6),
     Node([], 5),
     Node([], -5),
     Node([], 1),
     Node([], -3),
     Node([], 4),
     Node([], 1),
     Node([], -6),
     Node([], 7),
     Node([], 2),
     Node([], 8),
     Node([], -2),
     Node([], -1),
     Node([], 3),
     Node([], 0),
     Node([], 2),
     Node([], 5),
     Node([], -1),
     Node([], 4),
     Node([], -3),
     Node([], 5),
     Node([], 1),
     Node([], 7),
     Node([], -3),
     Node([], -1)]

n2 = [Node([n[0], n[1], n[2]], name="j"),
      Node([n[3], n[4], n[5]], name="k"),
      Node([n[6]], name="l"),
      Node([n[7], n[8]], name="m"),
      Node([n[9], n[10], n[11]], name="n"),
      Node([n[12], n[13]], name="o"),
      Node([n[14]], name="p"),
      Node([n[15], n[16]], name="q"),
      Node([n[17], n[18], n[19]], name="r"),
      Node([n[20], n[21]], name="s"),
      Node([n[22], n[23]], name="t"),
      Node([n[24], n[25], n[26]], name="u")]

n3 = [Node([n2[0], n2[1]], name="d"),
      Node([n2[2], n2[3]], name="e"),
      Node([n2[4]], name="f"),
      Node([n2[5], n2[6], n2[7]], name="g"),
      Node([n2[8], n2[9]], name="h"),
      Node([n2[10], n2[11]], name="i")]

n4 = [Node([n3[0], n3[1], n3[2]], name="b"),
      Node([n3[3], n3[4], n3[5]], name="c")]

n5 = Node([n4[0], n4[1]], name="a")

s = ALPHA_BETA_SEARCH(Game(), n5)
print(F"Pick move {s}")