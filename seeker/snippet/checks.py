#date: 2022-06-24T16:57:44Z
#url: https://api.github.com/gists/834194dc9490ddb6feab241d24c1e9c9
#owner: https://api.github.com/users/vmunoz82

from z3 import *

# Assume 32 bit ints
MIN_INT = -2147483648
MAX_INT = 2147483647

# Tools
upper  = lambda start, inc, iter, limit: (start+inc*iter) > limit
lower  = lambda start, inc, iter, limit: (start+inc*iter) < limit
upperi = lambda start, inc, iter, limit: (start+inc*iter) >= limit
loweri = lambda start, inc, iter, limit: (start+inc*iter) <= limit
top    = lambda start, inc, iter: upper(start, inc, iter, MAX_INT)
bottom = lambda start, inc, iter: lower(start, inc, iter, MIN_INT)
int_range = lambda v: And(v>=MIN_INT, v<=MAX_INT)

def intdiv(v, d):
  r = input % d
  nr = (input + (d - 1)) % d
  return If(input < 0, (input + (d - 1) - nr) / d, (input - r) / d)


# Checks
input = Int('input')  
k = intdiv(input, 100)
x = 2
y = k + 5

# Checks for overflows in lines 10-14 loop
i = Int('i')
# checks x bounds across iterations, this chek for an infinite loop
x_overflow = And(loweri(x, 1, i-1, 10), top(x, 1, i))
# checks for y int overflow
y_overflow = And(loweri(x, 1, i, 10), int_range(input), top(y, 3, i))

solve(x_overflow)
solve(y_overflow)

# Elided loop code
x = x + 8
y = y + 8 * 3

# Check division for zero in line 19
reachability = (3 * k + 100) > 43
zero_div = x == (y + 1)

solve(reachability)
solve(And(reachability, zero_div)) # to check by zero, we need reachability

