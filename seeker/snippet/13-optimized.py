#date: 2022-12-13T16:47:38Z
#url: https://api.github.com/gists/a155b21346dbc9faab5e4a4098d94c2e
#owner: https://api.github.com/users/maksverver

from functools import cmp_to_key
import sys

def ParseLine(line):
  stack = [[]]
  for ch in line:
    if ch == '[':
      stack.append([])
    elif ch == ']':
      stack[-2].append(stack.pop())
    elif ch == ',':
      stack[-1].append(0)
    else:
      assert ch.isdigit()
      if not stack[-1]:
        stack[-1].append(0)
      stack[-1][-1] = stack[-1][-1] * 10 + int(ch)
  assert len(stack) == 1
  return stack[0]

def ParsePart(part):
  a, b = part.splitlines()
  return ParseLine(a), ParseLine(b)

pairs = list(map(ParsePart, sys.stdin.read().split('\n\n')))


def Compare(a, b):
  if isinstance(a, int) and isinstance(b, int):
    return a - b

  if isinstance(a, int):
    return Compare([a], b)

  if isinstance(b, int):
    return Compare(a, [b])

  assert isinstance(a, list) and isinstance(b, list)
  for x, y in zip(a, b):
    if c := Compare(x, y):
      return c

  return len(a) - len(b)


def SolvePart1():
  for a, b in pairs:
    if Compare(a, b) == 0:
      print(a)
      print(b)
      assert False
  return sum(i for i, (a, b) in enumerate(pairs, 1) if Compare(a, b) < 0)


def SolvePart2():
  marker1 = [[2]]
  marker2 = [[6]]
  lists = [list for pair in pairs for list in pair]
  i1 = sum(Compare(list, marker1) < 0 for list in lists)
  i2 = sum(Compare(list, marker2) < 0 for list in lists)
  return (i1 + 1) * (i2 + 2)

sys.setrecursionlimit(1000000)
print(SolvePart1())
print(SolvePart2())
sys.exit()
