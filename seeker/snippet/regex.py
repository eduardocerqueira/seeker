#date: 2022-02-01T17:03:01Z
#url: https://api.github.com/gists/acabdc53c90e6f71aefa0d93878bdb28
#owner: https://api.github.com/users/affirVega

from copy import deepcopy

class BaseNode:
  star: bool

class AndNode:
  __match_args__ = ("nodes", "star")
  def __init__(self, nodes, star = False):
    self.nodes = nodes
    self.star = star

  def __repr__(self):
    return f'AndNode({self.nodes},{self.star})'

class OrNode:
  __match_args__ = ("nodes", "star")
  def __init__(self, nodes, star = False):
    self.nodes = nodes
    self.star = star

  def __repr__(self):
    return f'OrNode({self.nodes},{self.star})'

class CharNode:
  __match_args__ = ("char", "star")
  def __init__(self, char, star = False):
    self.char = char
    self.star = star
  
  def __repr__(self):
    return f'CharNode({self.char},{self.star})'

class Automata:
  def __init__(self):
    self.state_count = 0
    self.states = []
    self.actions = []
    self.table = {}
    self.start = None
    self.accepted = []

  def add(self, state, action, next):
    if state not in self.states:
      self.states.append(state)
    if next not in self.states:
      self.states.append(next)
    if action not in self.actions:
      self.actions.append(action)
    if state not in self.table:
      self.table[state] = {}
    if action not in self.table[state]:
      self.table[state][action] = []
    if next not in self.table[state][action]:
      self.table[state][action].append(next)
  
  def get(self, state, action) -> list | None:
    if state in self.table:
      if action in self.table[state]:
        return self.table[state][action]
    return None

  def next_state(self):
    self.state_count += 1
    return self.state_count

  def __str__(self):
    self.states.sort()
    self.actions.sort()
    s = ''
    s += 'States:  ' + ','.join(map(str, self.states)) + '\n'
    s += 'Actions: ' + ','.join(map(str, self.actions)) + '\n'
    
    max_len = 0
    for state in self.states:
      for action in self.actions:
        l = self.get(state, action)
        if l is not None:
          max_len = max(max_len, len(','.join(map(str, l))))
    
    s += '    ' + (''.join([str(a).center(max_len + 2) for a in self.actions])) + '\n'
    for state in self.states:
      s += '>' if state == self.start else ' '
      s += '*' if state in self.accepted else ' '
      s += f'{state:2}: '
      for action in self.actions:
        l = self.get(state, action)
        if l is None:
          s += '-'.center(max_len + 2)
        else:
          s += ','.join(map(str, l)).center(max_len + 2)
      s += '\n'
    return s

def to_dka(a: Automata):
  pass # todo

def to_automata(root: BaseNode, a: Automata = Automata()):
  a.start = 0
  ends = mkindex(root, a, [a.start])
  a.accepted += ends
  return a

def mkindex(root: BaseNode, a: Automata, starts: list[int], trans: list[tuple[str, int]] = None):
  match root:
    case AndNode(nodes, star):
      if star:
        transs = []
        ends = mkindex(nodes[0], a, starts, transs)
        for i in range(1, len(nodes)):
          ends = mkindex(nodes[i], a, ends)
        for tran in transs:
          for end in ends:
            a.add(end, tran[0], tran[1])
        if trans is not None: trans += transs
        return starts + ends
      
      ends = mkindex(nodes[0], a, starts, trans)
      for i in range(1, len(nodes)):
        ends = mkindex(nodes[i], a, ends)
      return ends
    case OrNode(nodes, star):
      ends = []
      transs = []
      for node in nodes:
        ends += mkindex(node, a, starts, transs)
      if trans is not None: trans += transs
      if star:
        for tran in transs:
          for end in ends:
            a.add(end, tran[0], tran[1])
        return starts + ends
      return ends
    case CharNode(char, star):
      new = a.next_state()
      for s in starts:
        a.add(s, char, new)
      if trans is not None:
        for s in starts:
          trans.append( (char, new) )
      if star:
        a.add(new, char, new)
        return [new] + starts
      return [new]

def to_tree(s: str, continue_index = None) -> BaseNode:
  ors: list[BaseNode] = []
  ands: list[BaseNode] = []
  i = 0
  while i < len(s):
    c = s[i]
    if c == '(':
      print('branch', i)
      a = []
      ands.append(to_tree(s[i + 1:], a))
      i += a[0] + 1
      print('new i', i)
    elif c == ')':
      if continue_index is not None:
        print('continue', i)
        continue_index.append(i)
      break
    elif c == '|':
      ors.append(AndNode(ands))
      ands = []
    elif c == '*':
      if len(ands) == 0:
        ors[-1].star = True
      else:
        ands[-1].star = True
    elif c == '\\':
      i += 1
      ands.append(CharNode(s[i]))
    else:
      ands.append(CharNode(c))
    
    i += 1
  
  if len(ors) == 0:
    return AndNode(ands)
  if len(ands) == 0:
    return OrNode(ors)
  if len(ands) == 1:
    ors.append(ands[0])
  else:
    ors.append(AndNode(ands))
  return OrNode(ors)

def print_tree(root: BaseNode, indent = 0):
  match root:
    case AndNode(nodes, star):
      print(' ' * indent, end='')
      print('and', end='')
      if star: print('*', end='')
      print()
      for node in nodes:
        print_tree(node, indent + 1)
        print()
    case OrNode(nodes, star):
      print(' ' * indent, end='')
      print('or', end='')
      if star: print('*', end='')
      print()
      for node in nodes:
        print_tree(node, indent + 1)
        print()
    case CharNode(char, star):
      print(' ' * indent, end='')
      print(char, end='')
      if star: print('*', end='')

tree = to_tree("a|b*a*|c|d*c")

a = to_automata(tree)

print(a)