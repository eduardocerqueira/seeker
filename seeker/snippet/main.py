#date: 2023-03-27T16:43:39Z
#url: https://api.github.com/gists/2775f0040ed6e1eb524d2925e5e45141
#owner: https://api.github.com/users/aklerza

arr = list(map(int, input().split()))
# author: aklerza

def s10():
    s=0
    for i in arr:
        if i < 0:
            s+=i
    return s

def s11():
    s=1
    for i in arr:
        if i>0:
            s*=i
    return s

def s12():
    a=int(input("a-nı daxil edin: "))
    s=0
    for i in arr:
        if i < a:
            s+=1
    return s

def s13():
    a=int(input("a-nı daxil edin: "))
    b=int(input("b-ni daxil edin: "))
    ans = arr
    for i in range(len(ans)):
        if ans[i] < 0:
            ans[i] += a
        else:
            ans[i] -= b
    return ans
    
def s14():
    a=int(input("a-nı daxil edin: "))
    s=0
    for i in arr:
        if i > a:
            s+=1
    return s

def s15():
    s=0
    for i in range(1, len(arr), 2): #indeksləmə əməliyyatı indeks 1-dən başlayarsa keçərlidir.
        s+= arr[i]
    return s

def s16():
    k = int(input("k-nı daxil edin: ")) - 1
    m = int(input("m-i daxil edin: ")) - 1
    i = m
    ans = []
    while i>=k:
        ans.append(arr[i])
        i-=1
    return ans

def s17():
    s=1
    for i in range(0, len(arr), 2): #indeksləmə əməliyyatı indeks 1-dən başlayarsa keçərlidir.
        s*= arr[i]
    return s

def s18():
    return max(arr) - min(arr)

def s19():
    m = int(len(arr)/2)
    cms = arr[m:]
    ans = arr
    maxs = cms.index(max(cms))
    ans[0], ans[m+maxs] = ans[m+maxs], ans[0]
    return ans

def s20():
    import math
    k = int(input("k-nı daxil edin: "))
    m = int(input("m-i daxil edin: "))    
    ans = 1
    for i in range(k,m+1):
        ans *= arr[i]
    return math.sqrt(ans)

def s21():
    return arr.count(min(arr))
    
def s22():
    return arr.count(max(arr))
    
def s23():
    arr[arr.index(min(arr))], arr[arr.index(max(arr))] = arr[arr.index(max(arr))], arr[arr.index(min(arr))]

def s24():
    for ind, val in enumerate(arr):
        if val > 0:
            arr[int(len(arr)-1)], arr[ind] = ar[ind], arr[int(len(arr)-1)]

def s25():
    arr[0], arr[int(len(arr)-1)] = arr[int(len(arr)-1)], arr[0]
    return arr

def s26():
    s=0
    for i in arr:
        s+=i
    return s / len(arr)

def s27():
    s=0
    k = int(input("k-nı daxil edin: "))
    m = int(input("m-i daxil edin: "))    
    for i in range(k,m+1):
        s+=arr[i]
    return s / m-k