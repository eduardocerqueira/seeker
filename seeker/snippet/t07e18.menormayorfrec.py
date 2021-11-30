#date: 2021-11-30T17:12:30Z
#url: https://api.github.com/gists/d3e4d00ceb046404592e03b9ea81817b
#owner: https://api.github.com/users/juanfal

# t07e18.menormayorfrec.py
# juanfc 2021-11-30
# 

def contar(l, x):
    cnt = 0
    for ele in l:
        if ele == x:
            cnt += 1
    return cnt


def menMayFrec(l):
    m = M = contar(l, l[0])
    for x in l:
        temp = contar(l, x)
        if  temp > M:
            M = temp
        elif temp < m:
            m = temp
    return m, M

# ----------------------

print(menMayFrec([1,2,3,2,1,3,1]))


