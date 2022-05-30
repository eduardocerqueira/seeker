#date: 2022-05-30T17:08:50Z
#url: https://api.github.com/gists/8e3e6c143758971564aafdace9625a92
#owner: https://api.github.com/users/tomhebbron

def rightplace(a,b,len=3):
    astr = str.zfill(str(a),len)
    bstr = str.zfill(str(b),len)
    return [1 if astr[x]==bstr[x] else 0 for x in range(len)]

def wrongplace(a,b,len=3):
    astr = str.zfill(str(a),len)
    bstr = str.zfill(str(b),len)
    return [1 if astr[x] in bstr and astr[x] != bstr[x] else 0 for x in range(len)]

for i in range(0,1000,1):
    ## 682: One number is correct and well placed.
    if sum(rightplace(i,682)) != 1: continue
    ## 614: One number is correct but wrongly placed.
    if sum(wrongplace(i,614)) != 1: continue
    ## 206: Two numbers are correct but wrongly placed.
    if sum(wrongplace(i,206)) != 2: continue
    ## 738: Nothing is correct.
    if sum(wrongplace(i,738)) != 0 or sum(rightplace(i,738)) != 0: continue
    ## 780: One number is correct but wrongly placed.
    if sum(wrongplace(i,380)) != 1: continue
    print(str.zfill(str(i),3))

