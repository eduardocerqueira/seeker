#date: 2021-11-19T17:05:27Z
#url: https://api.github.com/gists/53f4dd2d827371e894535b32074ec9e3
#owner: https://api.github.com/users/garthgilmour

def sample():
    print("sample")
    return 123


var1 = sample()
var2 = sample
var3 = {sample(): sample}

var2()
var3[var1]()
