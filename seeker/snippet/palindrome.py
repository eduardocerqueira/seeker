#date: 2022-06-20T16:59:43Z
#url: https://api.github.com/gists/796c0917e9ef415f9c0f083804441697
#owner: https://api.github.com/users/HousniBouchen

n = input("Type a string: ")
N=len(n)-1
i=0
while i<=len(n)/2:
    if n[i] != n[N]:
        break
    i=i+1
    N=N-1

if i > len(n)/2:
    print(n," is palindrome.\n")
else:
    print(n, " is not palindrome.\n")