#date: 2023-12-14T16:57:52Z
#url: https://api.github.com/gists/301bdc72dbe39b7eca54aea37521aa15
#owner: https://api.github.com/users/nzec

import sys
import callgraph

@callgraph.graphme('m', 'n')
def A(m, n):
    if m == 0:
        return n + 1
    if n == 0:
        return A(m - 1, 1)
    n2 = A(m, n - 1)
    return A(m - 1, n2)

print(A(3,2))
callgraph.view()
