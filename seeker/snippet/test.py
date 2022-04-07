#date: 2022-04-07T16:53:29Z
#url: https://api.github.com/gists/43d1fc54ceca91de0a64b6c24fac3191
#owner: https://api.github.com/users/iVoider

class Node:
   def __init__(self, data):
      self.data = data
      self.next = None
      self.prev = None
      self.pair = None
      self.completed = False

class doubly_linked_list:
   def __init__(self):
      self.head = None
      self.tail = None
      self.cur = None

   def push(self, data):     
        newNode = Node(data);     
        if self.head == None:     
            self.head = self.tail = newNode    
            self.head.previous = None   
            self.tail.next = None
        else:      
            self.tail.next = newNode
            self.tail.pair = newNode
            newNode.previous = self.tail    
            self.tail = newNode 
            self.tail.next = None
        self.cur = self.head

   def pop(self):
      if self.cur == self.tail:
        return None
      elif self.cur.completed == True:
        self.cur.pair = self.cur.pair.next
        self.cur.completed = False
      p = (self.cur.data, self.cur.pair.data)
      if self.cur.pair != self.tail:
        self.cur.pair = self.cur.pair.next
      else:
        self.cur.completed = True
        self.cur = self.cur.next
      return p

def tetrahedral(n):
    return (n * (n + 1) * (n + 2)) // 6

def upperbound(varcount, solvable):
  return tetrahedral(varcount - 2) * 7 if solvable else tetrahedral(varcount - 2) * 8
  
def children(p,p1,literals):
  general = set(p) | set(p1)
  exclude = set()
  for a,b in itertools.combinations(general,2):
    if a == b * -1:
      general.remove(a)
      general.remove(b)
      exclude.add(a)
      exclude.add(b)

  size = len(general)
  exsize = len(exclude)
  if size == 3 and exsize == 2:
    return { tuple(sorted(general, key = abs)) }
  elif size == 2 and exsize == 2:
    for e in general:
     exclude.add(e * -1)
    options = set()
    for l in literals - general - exclude:
     general.add(l)
     options.add(tuple(sorted(general,key=abs)))
     general.remove(l)
    return options
  return set()

def isat(formula,N):
   lim = upperbound(N,True)
   cursize = len(formula)
   literals = set(range(-1 * N,N + 1)) - {0}
   pairs = doubly_linked_list()
   
   for clause in formula:
     pairs.push(clause)

   while True:
      pair = pairs.pop()
      if pair != None:
       a,b = pair
       for child in children(a,b,literals):
        if child not in formula:
         pairs.push(child)
         formula.add(child)
       if len(formula) > lim:
        return False
      else:
        return True
   return True