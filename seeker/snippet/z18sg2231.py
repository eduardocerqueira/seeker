#date: 2022-02-14T17:01:18Z
#url: https://api.github.com/gists/04aeca5d7345355202385d96447d5c9f
#owner: https://api.github.com/users/alexei-math

from math import sqrt

# table = [13, 8, 69, 50, 30, 35, 57, 17, 32, 90, 55, 32, 44, 12, 80, 43]
table = []

f = open('18.csv', 'r')

line = f.readline()
while line:
   for n in line.split(','):
       table.append(int(n))
   line = f.readline()

# charge_cells = (5,)
# deny_cells = (10,)

charge_cells=(47, 57, 182, 192,)
deny_cells = (79, 85, 154, 160,)

table_max = []
table_min = []
N = int(sqrt(len(table)))

for i in range(N**2):
    table_max.append(0)
    table_min.append(0)

start_charge = 3000

table_max[0] = start_charge - table[0]
table_min[0] = start_charge - table[0]

for i in range(1, N):
    table_max[i] = table_max[i-1] - table[i]
    table_min[i] = table_min[i-1] - table[i]
    table_max[N*i] = table_max[N*(i-1)] - table[N*i]
    table_min[N*i] = table_min[N*(i-1)] - table[N*i]

for k in range(1, N):
    for i in range(1, N):
        if N*k+i not in deny_cells:
            if N*(k-1) + i not in deny_cells:
                v_vert = table_max[N*(k-1)+i] - table[N*k+i] if N*k+i not in charge_cells else table_max[N*(k-1)+i] + table[N*k+i]
            else:
                v_vert = -1000000
            if N*k+i-1 not in deny_cells:
                v_hor = table_max[N*k+i-1] - table[N*k+i] if N*k+i not in charge_cells else table_max[N*k+i-1] + table[N*k+i]
            else:
                v_hor = -1000000
            table_max[N*k+i] = max(v_vert, v_hor)
            if N*(k-1) + i not in deny_cells:
                v_vert = table_min[N*(k-1)+i] - table[N*k+i] if N*k+i not in charge_cells else table_min[N*(k-1)+i] + table[N*k+i]
            else:
                v_vert = 1000000
            if N*k+i-1 not in deny_cells:
                v_hor = table_min[N*k+i-1] - table[N*k+i] if N*k+i not in charge_cells else table_min[N*k+i-1] + table[N*k+i]
            else:
                v_hor = 1000000
            table_min[N*k+i] = min(v_vert, v_hor)
        else:
            table_max[N*k+i] = None
            table_min[N*k+i] = None

print(table_max[-1], table_min[-1])
