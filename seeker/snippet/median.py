#date: 2022-02-02T17:03:15Z
#url: https://api.github.com/gists/7a7241b26a5c62838c551b03d172e84f
#owner: https://api.github.com/users/erbenpeter

DEBUG = False


def dprint(*args):
  if DEBUG: print(*args)


class Slice:
  def __init__(self, l, low, high, big=False):
    self.l = l
    self.low = low
    self.high = high
    self.big = big # To avoid equality

  def size(self):
    return self.high - self.low + 1

  def mid(self):
    return (self.low + self.high) // 2

  def cut(self, low, high):
    return Slice(self.l, low, high)

  def pos(self, idx):
    return idx - self.low + 1

  def at_pos(self, pos):
    return self.l[self.low + pos - 1]  

  def min(self):
    return self.l[self.low]

  def max(self):
    return self.l[self.high]

  def __str__(self):
    return str(self.l[self.low:self.high+1]) + f' {self.low} {self.high} {self.big}'


def get_separated(pos, low, high):
  # pos is 1-based
  ll = len(low)
  if pos <= ll:
    return low[pos - 1]
  else:
    return high[pos - ll - 1]


def make_guess(slice1, slice2):
  if slice1.size() > slice2.size():
    return (1, slice1.mid())
  else:
    return (2, slice2.mid())


def smaller(val, other_slice):
  if other_slice.size() < 1:
    return 0
  if val < other_slice.min() or (val == other_slice.min() and other_slice.big):
    return 0
  if val > other_slice.max() or (val == other_slice.max() and not other_slice.big):
    return other_slice.size()
  a, b = other_slice.low, other_slice.high
  while b - a > 1:
    c = (a + b) // 2
    cv = other_slice.l[c]
    if val < cv or (val == cv and other_slice.big):
      a , b = a, c
    else:
      a, b = c, b
  return other_slice.pos(a)


def rec(guess, pos, slice1, slice2):
  lid, g_idx = guess # g_idx is absolute index
  if lid == 1: slice1, slice2 = slice2, slice1 # guess is now in slice2, searching in slice 1
  val = slice2.l[g_idx]
  if slice1.size() < 1:
    return slice2.at_pos(pos)
  smaller_in_other = smaller(val, slice1)
  pos_here = slice2.pos(g_idx)
  guess_pos = smaller_in_other + pos_here
  if guess_pos == pos:
    return val
  elif guess_pos > pos:
    slice1, slice2 = slice1.cut(slice1.low, slice1.low + smaller_in_other - 1), slice2.cut(slice2.low, g_idx - 1)
    guess = make_guess(slice1, slice2)
    return rec(guess, pos, slice1, slice2) # same position
  else:
    slice1, slice2 = slice1.cut(slice1.low + smaller_in_other, slice1.high), slice2.cut(g_idx + 1, slice2.high)
    guess = make_guess(slice1, slice2)
    return rec(guess, pos - guess_pos, slice1, slice2) # in the upper half


def search(pos, slice1, slice2):
  guess = make_guess(slice1, slice2)
  return rec(guess, pos, slice1, slice2)


def get_at_pos(pos, l1, l2):
  if l1[-1] < l2[0]: return get_separated(pos, l1, l2)
  if l2[-1] < l1[0]: return get_separated(pos, l2, l1)
  return search(pos, Slice(l1, 0, len(l1) - 1), Slice(l2, 0, len(l2) - 1, big=True))


def median(l1, l2):
  S = len(l1) + len(l2)
  if S % 2 == 1:
    return get_at_pos(S // 2 + 1, l1, l2)
  else:
    x = get_at_pos( S // 2, l1, l2)
    y = get_at_pos( S // 2 + 1, l1, l2)
    return (x + y) / 2
  

T = int(input())
for t in range(T):
  l1 = [int(x) for x in input().split()]
  l2 = [int(x) for x in input().split()]
  print(median(l1, l2))